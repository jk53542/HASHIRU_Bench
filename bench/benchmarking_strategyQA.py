from gradio_client import Client
import pandas as pd
import json
import time
import os
from datetime import datetime
import re
from datasets import load_dataset

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX
from benchmark_trace_context import hashiru_trace_context_prefix

# Gradio client HTTP timeout (seconds). Default handler fetch + predict can exceed 30s when
# HASHIRU is busy (long CEO traces, semantic metrics, GPU). Tune via env if needed.
_GRADIO_TIMEOUT = float(os.environ.get("HASHIRU_BENCH_GRADIO_TIMEOUT", "300"))

# Optional pause between questions (seconds). Per-question Client() reconnect was removed —
# it re-hit /config with a ~30s default timeout and failed while the server was still busy.
_INTER_Q_SLEEP = float(os.environ.get("HASHIRU_BENCH_STRATEGYQA_INTER_QUESTION_SLEEP", "5"))


def _make_gradio_client(url: str) -> Client:
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

def load_strategyqa_data(split="train", num_samples=None):
    """
    Load StrategyQA dataset from HuggingFace (ChilleD version)
    """
    # Load the dataset - using ChilleD's version which has better structure
    dataset = load_dataset("ChilleD/StrategyQA", split=split)
    df = pd.DataFrame(dataset)
    
    if num_samples:
        df = df.sample(n=num_samples, random_state=42)
        
    return df

def benchmark_strategyqa(df, out_dir="strategyqa_results", num_questions=10):
    """
    Benchmark multiagent system on StrategyQA dataset
    """
    if df is None or len(df) == 0:
        print("No data available for benchmarking")
        return
    
    # Sample questions if we have more than requested
    if len(df) > num_questions:
        all_questions = df.sample(n=num_questions, random_state=42)
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
    
    for idx, (i, row) in enumerate(all_questions.iterrows()):
        start = time.time()
        question_number = idx + 1
        question = row['question']
        
        # StrategyQA has boolean answers
        correct_answer = "yes" if row.get('answer', False) else "no"
        
        # Get reasoning steps if available
        facts = row.get('facts', []) if 'facts' in row else []
        
        # Create prompt for the multiagent system
        prompt_body = CEO_FORCE_AGENTS_PREFIX + "\n" + "You will be asked to answer strategic questions requiring multi-step thinking. " \
                "This question requires careful analysis and step-by-step reasoning. " \
                "Think through the problem logically and provide your final answer. " \
                "You MUST use agents. You may use tools only to support agents (e.g., retrieval), not as a replacement." \
                f"You have been asked the following question: {question} " \
                "Your answer must be either 'yes' or 'no'. " \
                "Reply with your answer in the format: {\"answer\":\"<YES_OR_NO>\"}. " \
                "The response should contain only this JSON format."
        
        max_retries = 3
        retry_count = 0
        agent_resp = None
        
        while retry_count < max_retries:
            try:
                trace_prefix = hashiru_trace_context_prefix(
                    benchmark_name="strategyqa",
                    question_index=question_number,
                    question_id=str(i),
                    bench_attempt=retry_count + 1,
                    question_text=question,
                )
                prompt = trace_prefix + prompt_body
                job = client.submit(
                    message={"text": prompt.strip(), "files": []},
                    api_name="/chat",
                )
                
                # Wait for completion
                while not job.done():
                    time.sleep(0.1)
                
                _preview, _history = job.outputs()[-1]
                agent_resp = extract_answer_from_chat_history(_history)
                
                if agent_resp:
                    break
                else:
                    retry_count += 1
                    prompt_body = f"The previous response did not follow the required format. " \
                            f"Please answer this question: {question} " \
                            f"Your answer must be either 'yes' or 'no' in the format: " \
                            f"{{\"answer\":\"<YES_OR_NO>\"}}. Please try again."
                    print(f"Invalid response, retrying ({retry_count}/{max_retries})")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Error during API call: {e}")
                retry_count += 1
                time.sleep(10)
        
        elapsed = time.time() - start
        
        # Prepare result
        result = {
            "question_num": question_number,
            "question": question,
            "correct_answer": correct_answer,
            "agent_resp": agent_resp,
            "is_correct": agent_resp == correct_answer if agent_resp else False,
            "time_elapsed": elapsed,
            "facts": facts if facts else [],
            "retry_count": retry_count
        }
        
        # Update score
        if agent_resp == correct_answer:
            correct_resp += 1
        
        total_processed += 1
        
        # Save result
        with open(out_path, "a") as f:
            f.write(json.dumps(result, indent=2) + "\n")
        
        # Print progress
        accuracy = (correct_resp / total_processed) * 100
        print(f"Question {question_number}/{num_questions} - "
              f"Score: {correct_resp}/{total_processed} ({accuracy:.1f}%) - "
              f"Time: {elapsed:.2f}s")

        if _INTER_Q_SLEEP > 0 and question_number < num_questions:
            time.sleep(_INTER_Q_SLEEP)
    
    # Final summary
    final_accuracy = (correct_resp / total_processed) * 100
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Questions: {total_processed}")
    print(f"Correct Answers: {correct_resp}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Results saved to: {out_path}")

def main():
    """
    Main function to run StrategyQA benchmark
    """
    print("Loading StrategyQA dataset...")
    
    # Load the dataset (you can change split to "train" if needed)
    df = load_strategyqa_data(split="test", num_samples=100)  # Load up to 100 samples
    
    if df is not None:
        print(f"Loaded {len(df)} questions from StrategyQA")
        print("Sample question:", df.iloc[0]['question'])
        print("Sample answer:", "yes" if df.iloc[0].get('answer', False) else "no")
        
        # Run benchmark
        benchmark_strategyqa(df=df, num_questions=100)
    else:
        print("Failed to load StrategyQA dataset")

if __name__ == "__main__":
    main()