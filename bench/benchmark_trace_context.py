"""
Prefix for Gradio /chat messages so HASHIRU orchestration JSONL includes benchmark fields.

Server strips ``[HASHIRU_TRACE_CTX]{...}\\n`` before the CEO sees the message; fields are
merged into every trace line for that user turn (``benchmark_name``, ``question_index``,
``question_id``, ``bench_attempt``).

Example::

    from benchmark_trace_context import hashiru_trace_context_prefix
    prompt = hashiru_trace_context_prefix(
        benchmark_name="strategyqa",
        question_index=1,
        question_id="row_42",
        bench_attempt=1,
    ) + user_visible_prompt
"""

from __future__ import annotations

import json

_PREFIX = "[HASHIRU_TRACE_CTX]"
_ALLOWED_KEYS = ("benchmark_name", "question_index", "question_id", "bench_attempt")


def hashiru_trace_context_prefix(
    *,
    benchmark_name: str,
    question_index: int,
    question_id: str | None = None,
    bench_attempt: int = 1,
) -> str:
    payload: dict = {
        "benchmark_name": benchmark_name,
        "question_index": int(question_index),
        "bench_attempt": int(bench_attempt),
    }
    if question_id is not None:
        payload["question_id"] = str(question_id)
    return _PREFIX + json.dumps(payload, separators=(",", ":")) + "\n"
