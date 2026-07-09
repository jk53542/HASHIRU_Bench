#!/usr/bin/env python3
r"""
Join an orchestration trace JSONL with a benchmark results file and print Markdown tables:
worker tool rounds (AskAgent + AskMultipleAgents per question) vs correct/incorrect.

The tables match the style used in prior notes, e.g.::

    | Worker calls | Correct | Incorrect | Total |
    |---:|---:|---:|---:|
    | 1 | ... | ... | ... |

Examples::

    # TruthfulQA (trace question_id = question_num - 1)
    python analyze_trace_worker_calls.py \\
        --bench truthful_qa \\
        --trace ../results/semantic_metrics_logs/trace_20260407_012530_12511.jsonl \\
        --results ../truthful_qa_results/truthful_qa_benchmark_20260407_012538.jsonl \\
        --out truthfulqa_worker_calls_table.md

    # JailbreakBench (one JSON object per line; question_id = sample_id)
    python analyze_trace_worker_calls.py \\
        --bench jailbreak \\
        --trace ../results/semantic_metrics_logs/trace_YYYYMMDD_*.jsonl \\
        --results ../results/jailbreakbench_benchmark_YYYYMMDD_*.jsonl \\
        --out jailbreak_worker_calls_table.md

Optional second table: accuracy split by whether AskMultipleAgents was used at least once.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_trace_worker_rounds(trace_path: Path) -> tuple[dict[str, int], dict[str, bool]]:
    """Per question_id: count of worker tool finishes; flag if any AskMultipleAgents."""
    counts: dict[str, int] = defaultdict(int)
    used_multi: dict[str, bool] = defaultdict(bool)
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("event") != "ceo_tool_finished":
                continue
            tool = o.get("tool")
            if tool not in ("AskAgent", "AskMultipleAgents"):
                continue
            qid = o.get("question_id")
            if qid is None:
                continue
            qk = str(qid)
            counts[qk] += 1
            if tool == "AskMultipleAgents":
                used_multi[qk] = True
    return dict(counts), dict(used_multi)


def load_truthfulqa_results(path: Path) -> dict[str, bool]:
    raw = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    idx = 0
    out: dict[str, bool] = {}
    L = len(raw)
    while idx < L:
        while idx < L and raw[idx].isspace():
            idx += 1
        if idx >= L:
            break
        obj, end = dec.raw_decode(raw, idx)
        idx = end
        qn = obj.get("question_num")
        if not isinstance(qn, int):
            continue
        qid = str(qn - 1)
        ev = obj.get("evaluation") or {}
        if "is_truthful" in ev:
            out[qid] = bool(ev["is_truthful"])
    return out


def load_jailbreak_results(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = o.get("sample_id")
            if sid is None:
                continue
            out[str(sid)] = bool(o.get("is_correct"))
    return out


def bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 4:
        return str(n)
    return "5+"


def build_table(
    rounds: dict[str, int],
    correct: dict[str, bool],
    correct_label: str,
    incorrect_label: str,
) -> str:
    # only question_ids present in both
    keys = sorted(set(rounds) & set(correct), key=lambda k: (len(k), k))
    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for qid in keys:
        b = bucket(rounds[qid])
        is_ok = correct[qid]
        if b == "0":
            continue
        if is_ok:
            rows[b][0] += 1
        else:
            rows[b][1] += 1
        rows[b][2] += 1

    order = ["1", "2", "3", "4", "5+"]
    lines = [
        f"| Worker calls | {correct_label} | {incorrect_label} | Total |",
        "|---:|---:|---:|---:|",
    ]
    tot_c = tot_i = tot_t = 0
    for b in order:
        c, i, t = rows.get(b, [0, 0, 0])
        lines.append(f"| {b} | {c} | {i} | {t} |")
        tot_c += c
        tot_i += i
        tot_t += t
    lines.append(f"| **All** | **{tot_c}** | **{tot_i}** | **{tot_t}** |")
    return "\n".join(lines)


def multi_table(correct: dict[str, bool], used_multi: dict[str, bool], rounds: dict[str, int]) -> str:
    keys = sorted(set(correct) & set(rounds), key=lambda k: (len(k), k))
    rows = {"multi": [0, 0, 0], "single_only": [0, 0, 0]}
    for qid in keys:
        if rounds[qid] <= 0:
            continue
        key = "multi_min1" if used_multi.get(qid) else "single_only"
        if key == "multi_min1":
            slot = "multi"
        else:
            slot = "single_only"
        is_ok = correct[qid]
        if is_ok:
            rows[slot][0] += 1
        else:
            rows[slot][1] += 1
        rows[slot][2] += 1

    lines = [
        "| Routing | Correct | Incorrect | Total |",
        "|---|---:|---:|---:|",
        f"| AskMultipleAgents used ≥1 | {rows['multi'][0]} | {rows['multi'][1]} | {rows['multi'][2]} |",
        f"| AskAgent only (no multi tool) | {rows['single_only'][0]} | {rows['single_only'][1]} | {rows['single_only'][2]} |",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", choices=["truthful_qa", "jailbreak"], required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="Write Markdown tables to this file (optional).")
    args = p.parse_args()

    rounds, used_multi = load_trace_worker_rounds(args.trace)
    if args.bench == "truthful_qa":
        correct = load_truthfulqa_results(args.results)
        ok_lab, bad_lab = "Truthful", "Untruthful"
    else:
        correct = load_jailbreak_results(args.results)
        ok_lab, bad_lab = "Correct (refusal)", "Incorrect"

    t1 = build_table(rounds, correct, ok_lab, bad_lab)
    t2 = multi_table(correct, used_multi, rounds)
    text = (
        f"# Worker call buckets\n\n"
        f"Trace: `{args.trace.name}`\n\n"
        f"Results: `{args.results.name}`  \n"
        f"Bench: `{args.bench}`\n\n"
        f"{t1}\n\n## Multi-agent vs single-tool routing\n\n{t2}\n"
    )
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nWrote {args.out}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
