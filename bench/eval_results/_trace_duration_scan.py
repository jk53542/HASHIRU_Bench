#!/usr/bin/env python3
"""Scan orchestration trace JSONL files: per-question duration and run totals."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "results" / "semantic_metrics_logs"
OUT = Path(__file__).resolve().parent / "_trace_duration_scan_out.json"


def analyze_trace(path: Path) -> dict | None:
    # Per question key: (benchmark_name, question_id_str)
    min_ts: dict[tuple[str, str], float] = {}
    max_final_ts: dict[tuple[str, str], float] = {}
    all_ts_bench: list[float] = []
    bench_rows = 0

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = o.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        bn = o.get("benchmark_name")
        if not bn or not isinstance(bn, str):
            continue
        bench_rows += 1
        all_ts_bench.append(float(ts))
        qid = o.get("question_id")
        if qid is None:
            continue
        qk = (bn, str(qid))
        cur = min_ts.get(qk)
        if cur is None or ts < cur:
            min_ts[qk] = float(ts)
        if o.get("event") == "ceo_final_answer":
            mf = max_final_ts.get(qk)
            if mf is None or ts > mf:
                max_final_ts[qk] = float(ts)

    durations: list[float] = []
    for qk, t0 in min_ts.items():
        t1 = max_final_ts.get(qk)
        if t1 is None or t1 < t0:
            continue
        durations.append(t1 - t0)

    if not durations and not all_ts_bench:
        return None

    wall = None
    if all_ts_bench:
        wall = max(all_ts_bench) - min(all_ts_bench)

    sum_dur = sum(durations) if durations else 0.0
    n = len(durations)
    mean = sum_dur / n if n else None
    med = statistics.median(durations) if n else None

    # dominant benchmark by count of completed questions
    by_bench: dict[str, int] = defaultdict(int)
    for qk, t0 in min_ts.items():
        t1 = max_final_ts.get(qk)
        if t1 is None or t1 < t0:
            continue
        by_bench[qk[0]] += 1
    dominant = max(by_bench, key=lambda b: by_bench[b]) if by_bench else None

    return {
        "file": path.name,
        "dominant_benchmark": dominant,
        "questions_completed": n,
        "per_question_durations_sum_s": round(sum_dur, 3),
        "wall_clock_bench_rows_s": round(wall, 3) if wall is not None else None,
        "mean_s": round(mean, 3) if mean is not None else None,
        "median_s": round(med, 3) if med is not None else None,
        "bench_row_count": bench_rows,
        "per_benchmark_completed": dict(by_bench),
    }


def main() -> None:
    traces = sorted(LOGS.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows: list[dict] = []
    for p in traces:
        r = analyze_trace(p)
        if r:
            rows.append(r)

    # Longest by sum of per-question durations (sequential benchmark proxy)
    by_sum = sorted(rows, key=lambda x: x["per_question_durations_sum_s"], reverse=True)
    by_wall = sorted(
        [x for x in rows if x.get("wall_clock_bench_rows_s") is not None],
        key=lambda x: x["wall_clock_bench_rows_s"],
        reverse=True,
    )

    OUT.write_text(
        json.dumps(
            {
                "trace_dir": str(LOGS),
                "count_traces_analyzed": len(rows),
                "top_by_sum_per_question": by_sum[:25],
                "top_by_wall_clock": by_wall[:25],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
