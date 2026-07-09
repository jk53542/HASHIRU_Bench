#!/usr/bin/env python3
"""Compare StrategyQA jsonl ablation runs (same 50-question slice)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "strategyqa_results"

RUNS: dict[str, Path] = {
    "both_095001": ROOT / "strategyqa_benchmark_20260414_095001.jsonl",
    "both_160235": ROOT / "strategyqa_benchmark_20260414_160235.jsonl",
    "neither_200850": ROOT / "strategyqa_benchmark_20260414_200850.jsonl",
    "density08_000224": ROOT / "strategyqa_benchmark_20260415_000224.jsonl",
    "density07_090706": ROOT / "strategyqa_benchmark_20260415_090706.jsonl",
    "density09_210431": ROOT / "strategyqa_benchmark_20260415_210431.jsonl",
    "entropy165_213252": ROOT / "strategyqa_benchmark_20260414_213252.jsonl",
    "entropy11_105654": ROOT / "strategyqa_benchmark_20260416_105654.jsonl",
}


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    idx = 0
    out: list[dict] = []
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        out.append(obj)
        idx = end
    return out


def main() -> None:
    data: dict[str, list[dict]] = {k: load_records(p) for k, p in RUNS.items()}
    for k, recs in data.items():
        acc = sum(1 for r in recs if r.get("is_correct")) / max(len(recs), 1)
        print(f"{k:22} n={len(recs):2} acc={acc:.3f}")

    keys = list(data.keys())
    base = "both_095001"
    b = data[base]
    if len(b) != 50:
        print("expected 50 in base")
        return

    def qkey(r: dict) -> str:
        return str(r.get("question", ""))[:120]

    for other in keys:
        if other == base:
            continue
        o = data[other]
        if len(o) != 50:
            continue
        diffs = []
        for i, (rb, ro) in enumerate(zip(b, o)):
            if qkey(rb) != qkey(ro):
                print("MISMATCH question text order", other, "at", i)
                break
            cb = bool(rb.get("is_correct"))
            co = bool(ro.get("is_correct"))
            if cb != co:
                # question_num is 1-based index in benchmark output
                qn = rb.get("question_num")
                diffs.append((qn, cb, co, rb.get("correct_answer"), rb.get("agent_resp"), ro.get("agent_resp")))
        wins = sum(1 for _, cb, co, *_ in diffs if (not cb) and co)
        losses = sum(1 for _, cb, co, *_ in diffs if cb and (not co))
        print(f"\nvs {other}: {len(diffs)} rows differ | base_wrong_other_right={wins} base_right_other_wrong={losses}")
        for row in diffs[:12]:
            print(" ", row)


if __name__ == "__main__":
    main()
