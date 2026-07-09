#!/usr/bin/env python3
"""
Map questions across two orchestration JSONL traces by normalized question_text, when
question_id is not shared between runs.

Usage:
  python3 _trace_question_text_matcher.py \\
    --trace-a ../results/semantic_metrics_logs/trace_20260406_003728_13082.jsonl \\
    --trace-b ../results/semantic_metrics_logs/trace_20260414_200844_3218330.jsonl \\
    --benchmark strategyqa

If --benchmark is omitted, any benchmark seen in both files is allowed for overlap.
Output: TSV on stdout: norm_hash\\tlen\\tquestion_id_a\\tquestion_id_b\\tsnippet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


def normalize(q: str) -> str:
    s = (q or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def first_qtext_events(path: Path) -> list[dict]:
    """First event per (benchmark, question_id) with question_text (any event type)."""
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("question_text")
            if not isinstance(t, str) or not t.strip():
                continue
            b = str(e.get("benchmark_name") or "")
            qid = str(e.get("question_id") or "")
            if not b or not qid:
                continue
            k = (b, qid)
            if k in seen:
                continue
            seen.add(k)
            rows.append(e)
    return rows


def build_index(
    path: Path, bench: str | None
) -> dict[str, list[tuple[str, str, str]]]:
    """
    key = normalized question_text
    value = list of (benchmark, question_id, trace_basename)
    """
    by_text: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    base = path.name
    for e in first_qtext_events(path):
        b = str(e.get("benchmark_name") or "")
        if bench and b != bench:
            continue
        t = normalize(e.get("question_text") or "")
        if not t:
            continue
        by_text[t].append((b, str(e.get("question_id")), base))
    return by_text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-a", type=Path, required=True)
    p.add_argument("--trace-b", type=Path, required=True)
    p.add_argument("--benchmark", type=str, default=None, help="e.g. strategyqa, gsm8k, mmlu_pro")
    args = p.parse_args()

    ia, ib = build_index(args.trace_a, args.benchmark), build_index(args.trace_b, args.benchmark)
    inter = set(ia) & set(ib)
    print(
        f"# trace_a={args.trace_a.name} trace_b={args.trace_b.name} "
        f"benchmark={args.benchmark or '*'}",
        file=sys.stderr,
    )
    print(
        f"# overlap_count={len(inter)} unique normalized question_text keys",
        file=sys.stderr,
    )
    for t in sorted(inter):
        ha = hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
        snip = t[:160] + ("…" if len(t) > 160 else "")
        pair_a, pair_b = ia[t], ib[t]
        for ba, qida, fa in pair_a:
            for bb, qidb, fb in pair_b:
                if args.benchmark and (ba != args.benchmark or bb != args.benchmark):
                    continue
                if not args.benchmark and ba != bb:
                    continue
                print(f"{ha}\t{len(t)}\t{qida}\t{qidb}\t{snip}")


if __name__ == "__main__":
    main()
