"""
Prefix for Gradio /chat messages so HASHIRU orchestration JSONL includes benchmark fields.

Server strips ``[HASHIRU_TRACE_CTX]{...}\\n`` before the CEO sees the message; fields are
merged into every trace line for that user turn (``benchmark_name``, ``question_index``,
``question_id``, ``bench_attempt``, optional ``question_text`` for offline alignment).

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
_ALLOWED_KEYS = (
    "benchmark_name",
    "question_index",
    "question_id",
    "bench_attempt",
    "question_text",
)


def hashiru_trace_context_prefix(
    *,
    benchmark_name: str,
    question_index: int,
    question_id: str | None = None,
    bench_attempt: int = 1,
    question_text: str | None = None,
) -> str:
    payload: dict = {
        "benchmark_name": benchmark_name,
        "question_index": int(question_index),
        "bench_attempt": int(bench_attempt),
    }
    if question_id is not None:
        payload["question_id"] = str(question_id)
    if question_text is not None and str(question_text).strip():
        qt = str(question_text).strip()
        if len(qt) > 2500:
            qt = qt[:2500]
        payload["question_text"] = qt
    return _PREFIX + json.dumps(payload, separators=(",", ":")) + "\n"
