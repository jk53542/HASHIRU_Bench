#!/usr/bin/env python3
"""Inventory benchmark jsonl/json + classify orchestration traces (both vs neither metrics)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVAL = Path(__file__).resolve().parent
LOGS = RESULTS / "semantic_metrics_logs"
BENCH = ROOT


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    idx = 0
    out: list[dict] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = dec.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        out.append(obj)
        idx = end
    return out


def load_jsonl_lines(path: Path) -> list[dict]:
    """One JSON object per line (paper_review, jailbreak)."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def classify_trace(path: Path) -> dict:
    """Infer semantic-metric mode from worker-tool finish rows."""
    rows = {"both_nn": 0, "both_null": 0, "e_only": 0, "d_only": 0, "other": 0}
    benches: set[str] = set()
    qids_by_bench: dict[str, set[str]] = defaultdict(set)

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            b = e.get("benchmark_name")
            if isinstance(b, str) and b.strip():
                benches.add(b)
                qid = e.get("question_id")
                if qid is not None:
                    qids_by_bench[b].add(str(qid))
            if e.get("event") != "ceo_tool_finished":
                continue
            if e.get("tool") not in ("AskAgent", "AskMultipleAgents"):
                continue
            if "semantic_entropy" not in e and "semantic_density" not in e:
                continue
            se, sd = e.get("semantic_entropy"), e.get("semantic_density")
            if se is None and sd is None:
                rows["both_null"] += 1
            elif se is not None and sd is not None:
                rows["both_nn"] += 1
            elif se is not None:
                rows["e_only"] += 1
            else:
                rows["d_only"] += 1

    total = sum(rows.values()) or 1
    if rows["both_nn"] / total >= 0.25:
        mode = "both_metrics_likely"
    elif rows["both_null"] / total >= 0.55:
        mode = "neither_metrics_likely"
    elif rows["e_only"] / total >= 0.35:
        mode = "entropy_only_likely"
    elif rows["d_only"] / total >= 0.35:
        mode = "density_only_likely"
    else:
        mode = "mixed_or_sparse"

    return {
        "file": path.name,
        "mode": mode,
        "metric_rows": dict(rows),
        "benchmarks": sorted(benches),
        "qid_counts": {b: len(s) for b, s in sorted(qids_by_bench.items())},
    }


def acc_bool(recs: list[dict], key: str = "is_correct") -> tuple[int, int, float | None]:
    ok = sum(1 for r in recs if r.get(key) is True)
    n = len(recs)
    return ok, n, (ok / n if n else None)


def main() -> None:
    lines: list[str] = []
    lines.append("# Benchmark run inventory (auto-generated)\n\n")

    # --- Traces ---
    traces = sorted(LOGS.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    lines.append("## Orchestration traces (semantic mode inferred)\n\n")
    lines.append(
        "Inference rule on `ceo_tool_finished` rows for AskAgent/AskMultipleAgents: "
        "count rows where both `semantic_entropy` and `semantic_density` keys exist; "
        "if ≥25% have both numeric → `both_metrics_likely`; if ≥55% have both null → `neither_metrics_likely`; "
        "else entropy-only / density-only / mixed.\n\n"
    )
    for p in traces[:60]:
        try:
            info = classify_trace(p)
        except OSError:
            continue
        lines.append(f"- `{info['file']}` → **{info['mode']}** | benchmarks={info['benchmarks']}\n")
        if info["qid_counts"]:
            lines.append(f"  - unique question_ids per benchmark: {info['qid_counts']}\n")
        lines.append(f"  - metric row counts: {info['metric_rows']}\n")

    # --- StrategyQA ---
    lines.append("\n## StrategyQA (`strategyqa_results/*.jsonl`)\n\n")
    sq = sorted((BENCH / "strategyqa_results").glob("strategyqa_benchmark_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in sq[:15]:
        recs = load_jsonl(p)
        ok, n, a = acc_bool(recs)
        accs = f"{a:.4f}" if a is not None else "n/a"
        lines.append(f"- `{p.name}` n={n} acc={accs}\n")

    # --- TruthfulQA ---
    lines.append("\n## TruthfulQA (`truthful_qa_results/*.jsonl`)\n\n")
    tq = sorted((BENCH / "truthful_qa_results").glob("truthful_qa_benchmark_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in tq[:12]:
        recs = load_jsonl(p)
        ok, n, a = acc_bool(recs)
        accs = f"{a:.4f}" if a is not None else "n/a"
        lines.append(f"- `{p.name}` n={n} acc={accs}\n")

    # --- Jailbreak ---
    lines.append("\n## JailbreakBench (`results/jailbreakbench_benchmark_*.jsonl`)\n\n")
    jb = sorted(RESULTS.glob("jailbreakbench_benchmark_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in jb[:8]:
        recs = load_jsonl_lines(p)
        ok, n, a = acc_bool(recs)
        accs = f"{a:.4f}" if a is not None else "n/a"
        lines.append(f"- `{p.name}` n={n} acc={accs}\n")

    # --- GSM8K ---
    lines.append("\n## GSM8K (`eval_results/gsm8k_test_*.jsonl`)\n\n")
    for p in sorted(EVAL.glob("gsm8k_test_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)[:6]:
        recs = load_jsonl(p)
        ok, n, a = acc_bool(recs)
        accs = f"{a:.4f}" if a is not None else "n/a"
        lines.append(f"- `{p.name}` n={n} acc={accs}\n")

    # --- Paper review ---
    lines.append("\n## Paper review (`results/paper_review_benchmark_*.jsonl`)\n\n")
    pr = sorted(RESULTS.glob("paper_review_benchmark_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pr[:10]:
        recs = load_jsonl_lines(p)
        ok, n, a = acc_bool(recs)
        sem = sum(1 for r in recs if r.get("semantic_metrics_called")) if recs else 0
        accs = f"{a:.4f}" if a is not None else "n/a"
        lines.append(f"- `{p.name}` n={n} acc={accs} semantic_metrics_called_true={sem}\n")

    # --- MMLU-Pro summaries ---
    lines.append("\n## MMLU-Pro per-subject summaries (`eval_results/*_summary.json`)\n\n")
    for p in sorted(EVAL.glob("*_summary.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # flatten acc
        subj_accs: list[tuple[str, float]] = []
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "acc" in v:
                    subj_accs.append((k, float(v["acc"])))
        if subj_accs:
            w = sum(v.get("wrong", 0) or 0 for v in d.values() if isinstance(v, dict))
            c = sum(v.get("corr", 0) or 0 for v in d.values() if isinstance(v, dict))
            macro = sum(a for _, a in subj_accs) / len(subj_accs) if subj_accs else None
            lines.append(
                f"- `{p.name}` subjects={len(subj_accs)} macro_acc≈{macro:.4f} total_corr={c} total_wrong={w}\n"
            )

    # --- tau2 summaries ---
    lines.append("\n## Tau2 summary jsonl (`results/tau2_*_summary*.jsonl`)\n\n")
    for p in sorted(RESULTS.glob("tau2_*summary*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)[:15]:
        recs = load_jsonl_lines(p)
        if not recs:
            continue
        first = recs[0]
        if first.get("error"):
            lines.append(f"- `{p.name}` ERROR: {first.get('error')}\n")
            continue
        # try common success shape
        lines.append(f"- `{p.name}` rows={len(recs)} sample_keys={list(first.keys())[:12]}\n")

    out = EVAL / "benchmark_runs_inventory.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
