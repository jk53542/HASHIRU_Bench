"""
Shared per-question wall-clock pacing for HASHIRU benchmark scripts.

Fast worker backends (e.g. ChatGPT) can finish long before slower CEO (Gemini) calls,
which may hurt rate limits or answer quality. After a question completes, sleep only
if needed so each question occupies at least ``min_seconds`` wall time.

Environment (read by callers that choose to wire this in):

  HASHIRU_BENCH_MIN_QUESTION_SECONDS — non-negative float; ``0`` disables pacing.
  Not set — callers supply their own default (e.g. StrategyQA defaults to 60).
"""
from __future__ import annotations

import time


def apply_min_question_floor(
    start: float, min_seconds: float
) -> tuple[float, float]:
    """
    If ``min_seconds`` > 0 and less than that many seconds have passed since
    ``start`` (as returned by ``time.time()``), sleep until the floor is met.

    Returns ``(elapsed_total_seconds, buffer_sleep_seconds)`` where elapsed is
    ``time.time() - start`` after any sleep.
    """
    if min_seconds <= 0:
        return time.time() - start, 0.0
    elapsed = time.time() - start
    if elapsed >= min_seconds:
        return elapsed, 0.0
    buf = min_seconds - elapsed
    time.sleep(buf)
    return time.time() - start, buf
