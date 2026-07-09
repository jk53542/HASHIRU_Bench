"""
Shared helpers for HASHIRU / HASSUM benchmark scripts.

Follows the StrategyQA pattern: Gradio client, trace context, worker-tool enforcement,
JSONL output, and optional per-question pacing.
"""

from __future__ import annotations

import json
import os
import re
import string
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Optional

from benchmark_timing import apply_min_question_floor
from benchmark_trace_context import hashiru_trace_context_prefix

_INTERNAL_TOOL_JSON_PREFIX = "hashiru-internal-json:"
_GRADIO_TIMEOUT = float(os.environ.get("HASHIRU_BENCH_GRADIO_TIMEOUT", "300"))
_INTER_Q_SLEEP = float(os.environ.get("HASHIRU_BENCH_INTER_QUESTION_SLEEP", "5"))


def min_question_seconds_from_env(default: float = 0.0) -> float:
    raw = os.environ.get("HASHIRU_BENCH_MIN_QUESTION_SECONDS", str(default)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return max(0.0, default)


def make_gradio_client(url: str):
    from gradio_client import Client

    try:
        return Client(url, httpx_kwargs={"timeout": _GRADIO_TIMEOUT})
    except TypeError:
        return Client(url)


def init_hashiru_client(url: Optional[str] = None):
    client = make_gradio_client(url or os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860/"))
    client.predict(
        modeIndexes=[
            "ENABLE_AGENT_CREATION",
            "ENABLE_LOCAL_AGENTS",
            "ENABLE_CLOUD_AGENTS",
            "ENABLE_TOOL_CREATION",
            "ENABLE_TOOL_INVOCATION",
            "ENABLE_RESOURCE_BUDGET",
            "ENABLE_ECONOMY_BUDGET",
        ],
        api_name="/update_model",
    )
    return client


def history_used_worker_tools(history) -> bool:
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


def _assistant_text_from_message(m: dict) -> str:
    content = m.get("content", "")
    if isinstance(content, (list, tuple)):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p.get("text", "")))
        return "\n".join(parts)
    return str(content)


def latest_assistant_text(history) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            return _assistant_text_from_message(m).strip()
    return ""


def extract_json_field(history, field: str, patterns: list[str]) -> Optional[str]:
    """Scan assistant messages (newest first) for a JSON field such as answer or choice."""
    if not history:
        return None
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        text = _assistant_text_from_message(m)
        if not text:
            continue
        lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lower, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return None


def extract_short_answer(history) -> Optional[str]:
    patterns = [
        r'\{"answer"\s*:\s*"([^"]+)"\s*\}',
        r'"answer"\s*:\s*"([^"]+)"',
        r"answer\s*:\s*['\"]?([^'\"}\n]+)['\"]?",
    ]
    return extract_json_field(history, "answer", patterns)


def extract_mcq_choice(history) -> Optional[str]:
    patterns = [
        r'\{"choice"\s*:\s*"([A-Da-d])"\s*\}',
        r'"choice"\s*:\s*"([A-Da-d])"',
        r'\bchoice\s*:\s*["\']?([A-Da-d])["\']?',
        r'\b(?:answer|option)\s*(?:is|:)\s*["\']?([A-Da-d])["\']?',
    ]
    val = extract_json_field(history, "choice", patterns)
    return val.upper() if val else None


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization for short free-form answers."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join([chr(8216), chr(8217), chr(180), chr(96)]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s or ""))))).strip()


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def make_output_path(out_dir: str, benchmark_slug: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{benchmark_slug}_benchmark_{timestamp}.jsonl")


def run_hashiru_turn(
    client,
    *,
    benchmark_name: str,
    question_index: int,
    question_id: str,
    question_text: str,
    prompt_body: str,
    bench_attempt: int = 1,
) -> tuple[str, list, bool]:
    trace_prefix = hashiru_trace_context_prefix(
        benchmark_name=benchmark_name,
        question_index=question_index,
        question_id=question_id,
        bench_attempt=bench_attempt,
        question_text=question_text,
    )
    job = client.submit(
        message={"text": (trace_prefix + prompt_body).strip(), "files": []},
        api_name="/chat",
    )
    while not job.done():
        time.sleep(0.1)
    _preview, history = job.outputs()[-1]
    used_workers = history_used_worker_tools(history)
    return latest_assistant_text(history), history, used_workers


def benchmark_loop(
    *,
    benchmark_name: str,
    benchmark_slug: str,
    items: list[dict],
    out_dir: str,
    build_prompt: Callable[[dict], str],
    parse_response: Callable[[list], Any],
    score_item: Callable[[dict, Any, bool], dict],
    require_worker_tools: bool = True,
    max_retries: int = 5,
    min_question_seconds: float = 0.0,
    retry_suffix_worker: Callable[[dict], str],
    retry_suffix_format: Callable[[dict], str],
    extra_result_fields: Optional[Callable[[dict], dict]] = None,
) -> dict:
    """
    Generic benchmark driver. Each ``item`` must include at least ``question_id`` and
    ``question`` (display text for traces).
    """
    out_path = make_output_path(out_dir, benchmark_slug)
    print(f"Writing results to {out_path}")

    client = init_hashiru_client()
    correct = 0
    total = 0
    n_items = len(items)

    if require_worker_tools:
        print(
            "Benchmark: require_worker_tools=True — rejecting runs with no AskAgent/AskMultipleAgents "
            "in Gradio chat history.\n"
            "For strongest enforcement, set HASHIRU_STRICT_WORKER_MANDATE=1 on the HASHIRU server."
        )

    for idx, item in enumerate(items):
        start = time.time()
        question_number = idx + 1
        question_id = str(item.get("question_id", idx))
        question_text = str(item.get("question", ""))

        parsed = None
        used_workers = False
        attempts_made = 0
        worker_extra = ""
        format_extra = ""

        for attempt in range(1, max_retries + 1):
            attempts_made = attempt
            try:
                prompt_body = build_prompt(item) + worker_extra + format_extra
                _text, history, used_workers = run_hashiru_turn(
                    client,
                    benchmark_name=benchmark_name,
                    question_index=question_number,
                    question_id=question_id,
                    question_text=question_text,
                    prompt_body=prompt_body,
                    bench_attempt=attempt,
                )
                parsed = parse_response(history)
                ok_workers = (not require_worker_tools) or used_workers
                ok_format = parsed is not None and parsed != ""
                if ok_format and ok_workers:
                    break
                if attempt >= max_retries:
                    break
                if require_worker_tools and not used_workers:
                    print(
                        f"No AskAgent/AskMultipleAgents in chat history; retrying ({attempt}/{max_retries})."
                    )
                    worker_extra = retry_suffix_worker(item)
                elif not ok_format:
                    print(f"Invalid or unparseable answer format, retrying ({attempt}/{max_retries})")
                    format_extra = retry_suffix_format(item)
                time.sleep(5)
            except Exception as e:
                print(f"Error during API call: {e}")
                if attempt >= max_retries:
                    break
                time.sleep(10)

        elapsed, min_q_buffer = apply_min_question_floor(start, min_question_seconds)
        mandate_violation = bool(require_worker_tools and not used_workers)
        score = score_item(item, parsed, mandate_violation)

        result = {
            "question_num": question_number,
            "question_id": question_id,
            "question": question_text,
            "agent_resp": parsed,
            "time_elapsed": elapsed,
            "retry_count": max(0, attempts_made - 1),
            "bench_attempts": attempts_made,
            "used_worker_tools": used_workers,
            "require_worker_tools": require_worker_tools,
            "worker_mandate_violation": mandate_violation,
            "min_question_buffer_seconds": min_q_buffer,
            "min_question_floor_seconds": min_question_seconds,
            **score,
        }
        if extra_result_fields:
            result.update(extra_result_fields(item))

        if score.get("is_correct"):
            correct += 1
        total += 1

        with open(out_path, "a") as f:
            f.write(json.dumps(result, indent=2) + "\n")

        accuracy = (correct / total) * 100 if total else 0.0
        buf_note = f" (+{min_q_buffer:.1f}s pacing)" if min_q_buffer > 0 else ""
        print(
            f"Question {question_number}/{n_items} - "
            f"Score: {correct}/{total} ({accuracy:.1f}%) - "
            f"Time: {elapsed:.2f}s{buf_note}"
        )

        if _INTER_Q_SLEEP > 0 and question_number < n_items:
            time.sleep(_INTER_Q_SLEEP)

    final_accuracy = (correct / total) * 100 if total else 0.0
    print("\n=== FINAL RESULTS ===")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Results saved to: {out_path}")
    return {
        "total_processed": total,
        "correct_count": correct,
        "accuracy": final_accuracy / 100.0,
        "output_file": out_path,
    }


def add_common_cli_args(parser) -> None:
    parser.add_argument("--num-questions", type=int, default=50, help="Questions to evaluate after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the split.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for JSONL outputs.")
    parser.add_argument(
        "--allow-ceo-only",
        action="store_true",
        help="Accept CEO answers even when no worker tool call appears in chat history.",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Max benchmark-side retries per question.")
    parser.add_argument(
        "--min-question-seconds",
        type=float,
        default=None,
        help="Minimum wall time per question (default: HASHIRU_BENCH_MIN_QUESTION_SECONDS or 0).",
    )
