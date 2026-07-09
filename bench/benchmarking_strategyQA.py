import pandas as pd
import json
import time
import os
import argparse
from datetime import datetime
import re

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_STRATEGYQA
from benchmark_trace_context import hashiru_trace_context_prefix
from benchmark_timing import apply_min_question_floor

# Matches HASHIRU manager internal serialization for tool calls in chat history.
_INTERNAL_TOOL_JSON_PREFIX = "hashiru-internal-json:"

# Gradio client HTTP timeout (seconds). Default handler fetch + predict can exceed 30s when
# HASHIRU is busy (long CEO traces, semantic metrics, GPU). Tune via env if needed.
_GRADIO_TIMEOUT = float(os.environ.get("HASHIRU_BENCH_GRADIO_TIMEOUT", "300"))

# Optional pause between questions (seconds). Per-question Client() reconnect was removed —
# it re-hit /config with a ~30s default timeout and failed while the server was still busy.
_INTER_Q_SLEEP = float(os.environ.get("HASHIRU_BENCH_STRATEGYQA_INTER_QUESTION_SLEEP", "5"))

# Minimum wall time per question (seconds), applied *after* HASHIRU finishes (including retries).
# If the turn took less than this, sleeps the remainder so fast worker APIs do not overwhelm
# slower Gemini CEO calls. Set HASHIRU_BENCH_MIN_QUESTION_SECONDS=0 to disable.
def _min_question_seconds_from_env() -> float:
    raw = os.environ.get("HASHIRU_BENCH_MIN_QUESTION_SECONDS", "60").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


_MIN_QUESTION_SECONDS = _min_question_seconds_from_env()


def _make_gradio_client(url: str):
    from gradio_client import Client

    try:
        return Client(url, httpx_kwargs={"timeout": _GRADIO_TIMEOUT})
    except TypeError:
        return Client(url)


def sanitize_response(input_str):
    """
    Extract yes/no answer from the response
    Handles various formats like: {"answer": "yes"}, answer: "no", etc.
    """
    if not input_str:
        return None
    # Convert to lowercase for case-insensitive matching
    input_lower = input_str.lower()

    # Try to match structured formats first
    patterns = [
        r'\{"answer"\s*:\s*"(yes|no)"\s*\}',
        r'"answer"\s*:\s*"(yes|no)"',
        r"'answer'\s*:\s*'(yes|no)'",
        r"answer\s*:\s*['\"]?(yes|no)['\"]?",
        r'\{"choice"\s*:\s*"(yes|no)"\s*\}',
        r'"choice"\s*:\s*"(yes|no)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, input_lower)
        if match:
            return match.group(1)

    # Fallback: look for explicit yes/no in the response
    if "yes" in input_lower and "no" not in input_lower:
        return "yes"
    elif "no" in input_lower and "yes" not in input_lower:
        return "no"

    return None


def extract_answer_from_chat_history(history):
    """Find the latest assistant message that contains a parseable yes/no (CEO may emit many tool steps)."""
    if not history:
        return None
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if isinstance(content, (list, tuple)):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(str(p.get("text", "")))
            content = "\n".join(parts)
        ans = sanitize_response(str(content))
        if ans:
            return ans
    return None


def history_used_worker_tools(history) -> bool:
    """
    True if the chat contains at least one AskAgent / AskMultipleAgents tool invocation
    (same tools enforced by HASHIRU worker mandate in manager.py).
    """
    if not history:
        return False
    blob = json.dumps(history, default=str)
    if '"name": "AskAgent"' in blob or '"name": "AskMultipleAgents"' in blob:
        return True
    for m in history:
        if m.get("role") != "function_call":
            continue
        c = m.get("content")
        if isinstance(c, list):
            for item in c:
                if not isinstance(item, dict):
                    continue
                if item.get("kind") != "function_call":
                    continue
                if item.get("name") in ("AskAgent", "AskMultipleAgents"):
                    return True
        if isinstance(c, str) and c.startswith(_INTERNAL_TOOL_JSON_PREFIX):
            try:
                dec = json.loads(c[len(_INTERNAL_TOOL_JSON_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if isinstance(dec, list):
                for item in dec:
                    if (
                        isinstance(item, dict)
                        and item.get("kind") == "function_call"
                        and item.get("name") in ("AskAgent", "AskMultipleAgents")
                    ):
                        return True
    return False


def load_strategyqa_data(split="train", num_samples=None, offset=0):
    """
    Load StrategyQA dataset from HuggingFace (ChilleD version)
    """
    from datasets import load_dataset

    # Load the dataset - using ChilleD's version which has better structure
    dataset = load_dataset("ChilleD/StrategyQA", split=split)
    df = pd.DataFrame(dataset)
    
    # Deterministic contiguous slice so ablations can run the exact same questions.
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if num_samples is not None:
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        df = df.iloc[offset : offset + num_samples]
    elif offset > 0:
        df = df.iloc[offset:]

    return df

def benchmark_strategyqa(
    df,
    out_dir="strategyqa_results",
    num_questions=10,
    *,
    require_worker_tools: bool = True,
    max_retries: int = 5,
):
    """
    Benchmark multiagent system on StrategyQA dataset
    """
    if df is None or len(df) == 0:
        print("No data available for benchmarking")
        return
    
    # Deterministic contiguous subset from the already-sliced dataframe
    if len(df) > num_questions:
        all_questions = df.iloc[:num_questions]
    else:
        all_questions = df
        num_questions = len(df)
    
    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"strategyqa_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")
    
    # Single client for the whole run (reconnecting after every question often timed out:
    # Client() re-fetches /config while HASHIRU was still busy from the prior turn.)
    try:
        client = _make_gradio_client(os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860/"))
        client.predict(
            modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS",
                        "ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET",
                        "ENABLE_ECONOMY_BUDGET"],
            api_name="/update_model"
        )
    except Exception as e:
        print(f"Error connecting to client: {e}")
        return
    
    correct_resp = 0
    total_processed = 0

    if require_worker_tools:
        print(
            "Benchmark: require_worker_tools=True — rejecting runs with no AskAgent/AskMultipleAgents "
            "in Gradio chat history.\n"
            "For strongest enforcement, set HASHIRU_STRICT_WORKER_MANDATE=1 on the HASHIRU server "
            "(see HASHIRU_modified/src/manager/manager.py)."
        )
    
    for idx, (i, row) in enumerate(all_questions.iterrows()):
        start = time.time()
        question_number = idx + 1
        question = row['question']
        
        # StrategyQA has boolean answers
        correct_answer = "yes" if row.get('answer', False) else "no"
        
        # Get reasoning steps if available
        facts = row.get('facts', []) if 'facts' in row else []
        
        base_task = (
            CEO_FORCE_AGENTS_PREFIX_STRATEGYQA
            + "\n"
            + "You will be asked to answer strategic questions requiring multi-step thinking. "
            + "This question requires careful analysis and step-by-step reasoning. "
            + "Think through the problem logically and provide your final answer. "
            + "You MUST use agents. You may use tools only to support agents (e.g., retrieval), not as a replacement. "
            + f"You have been asked the following question: {question} "
            + "Your answer must be either 'yes' or 'no'. "
            + 'Reply with your answer in the format: {"answer":"<YES_OR_NO>"}. '
            + "The response should contain only this JSON format."
        )
        worker_extra = ""
        format_extra = ""

        agent_resp = None
        used_workers = False
        attempts_made = 0

        for attempt in range(1, max_retries + 1):
            attempts_made = attempt
            try:
                trace_prefix = hashiru_trace_context_prefix(
                    benchmark_name="strategyqa",
                    question_index=question_number,
                    question_id=str(i),
                    bench_attempt=attempt,
                    question_text=question,
                )
                prompt = trace_prefix + base_task + worker_extra + format_extra
                job = client.submit(
                    message={"text": prompt.strip(), "files": []},
                    api_name="/chat",
                )
                
                # Wait for completion
                while not job.done():
                    time.sleep(0.1)
                
                _preview, _history = job.outputs()[-1]
                used_workers = history_used_worker_tools(_history)
                agent_resp = extract_answer_from_chat_history(_history)

                ok_workers = (not require_worker_tools) or used_workers
                ok_format = bool(agent_resp)

                if ok_format and ok_workers:
                    break

                if attempt >= max_retries:
                    break

                if require_worker_tools and not used_workers:
                    print(
                        f"No AskAgent/AskMultipleAgents in chat history; retrying ({attempt}/{max_retries}). "
                        "Set HASHIRU_STRICT_WORKER_MANDATE=1 on HASHIRU to discard CEO-only turns server-side."
                    )
                    worker_extra = (
                        "\n\n[Benchmark enforcement] Your previous completion did not show any AskAgent or "
                        "AskMultipleAgents tool call in the conversation. You must call GetAgents if needed, "
                        "then AskAgent or AskMultipleAgents, read the worker reply, and only then output the "
                        f'final JSON for: {question}'
                    )
                elif not ok_format:
                    print(f"Invalid or unparseable answer format, retrying ({attempt}/{max_retries})")
                    format_extra = (
                        "\n\n[Benchmark enforcement] The previous response did not follow the required format. "
                        f"Answer this question: {question} "
                        'Your answer must be either yes or no in the format: {"answer":"<YES_OR_NO>"}. '
                        "Remember: call AskAgent or AskMultipleAgents before the final JSON."
                    )
                time.sleep(5)
                    
            except Exception as e:
                print(f"Error during API call: {e}")
                if attempt >= max_retries:
                    break
                time.sleep(10)
        
        elapsed, min_q_buffer = apply_min_question_floor(start, _MIN_QUESTION_SECONDS)
        
        # Prepare result
        mandate_violation = bool(require_worker_tools and not used_workers)
        is_correct = (
            bool(agent_resp)
            and (agent_resp == correct_answer)
            and not mandate_violation
        )
        result = {
            "question_num": question_number,
            "question": question,
            "correct_answer": correct_answer,
            "agent_resp": agent_resp,
            "is_correct": is_correct,
            "time_elapsed": elapsed,
            "facts": facts if facts else [],
            "retry_count": max(0, attempts_made - 1),
            "bench_attempts": attempts_made,
            "used_worker_tools": used_workers,
            "require_worker_tools": require_worker_tools,
            "worker_mandate_violation": mandate_violation,
            "min_question_buffer_seconds": min_q_buffer,
            "min_question_floor_seconds": _MIN_QUESTION_SECONDS,
        }
        
        # Update score
        if is_correct:
            correct_resp += 1
        
        total_processed += 1
        
        # Save result
        with open(out_path, "a") as f:
            f.write(json.dumps(result, indent=2) + "\n")
        
        # Print progress
        accuracy = (correct_resp / total_processed) * 100
        _buf_note = (
            f" (+{min_q_buffer:.1f}s pacing)"
            if min_q_buffer > 0
            else ""
        )
        print(f"Question {question_number}/{num_questions} - "
              f"Score: {correct_resp}/{total_processed} ({accuracy:.1f}%) - "
              f"Time: {elapsed:.2f}s{_buf_note}")

        if _INTER_Q_SLEEP > 0 and question_number < num_questions:
            time.sleep(_INTER_Q_SLEEP)
    
    # Final summary
    final_accuracy = (correct_resp / total_processed) * 100
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Questions: {total_processed}")
    print(f"Correct Answers: {correct_resp}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Results saved to: {out_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StrategyQA benchmark with deterministic slicing.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of contiguous questions to evaluate after offset (default: 100).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index in the split for contiguous evaluation (default: 0).",
    )
    parser.add_argument(
        "--out-dir",
        default="strategyqa_results",
        help="Directory for jsonl benchmark outputs (default: strategyqa_results).",
    )
    parser.add_argument(
        "--allow-ceo-only",
        action="store_true",
        help=(
            "If set, accept CEO answers even when no AskAgent/AskMultipleAgents appears in chat history. "
            "Default is to require at least one worker tool call (recommended for ablations)."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max benchmark-side retries per question (format + worker-tool enforcement). Default: 5.",
    )
    return parser


def main():
    """
    Main function to run StrategyQA benchmark
    """
    args = build_parser().parse_args()
    print("Loading StrategyQA dataset...")
    
    df = load_strategyqa_data(
        split=args.split,
        num_samples=args.num_questions,
        offset=args.offset,
    )
    
    if df is not None:
        print(f"Loaded {len(df)} questions from StrategyQA")
        print("Sample question:", df.iloc[0]['question'])
        print("Sample answer:", "yes" if df.iloc[0].get('answer', False) else "no")
        
        # Run benchmark
        benchmark_strategyqa(
            df=df,
            out_dir=args.out_dir,
            num_questions=args.num_questions,
            require_worker_tools=not args.allow_ceo_only,
            max_retries=max(1, args.max_retries),
        )
    else:
        print("Failed to load StrategyQA dataset")

if __name__ == "__main__":
    main()