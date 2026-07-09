#!/usr/bin/env python3
"""
Find StrategyQA items where traces show semantic-driven follow-up and JSONL shows
correct answer while other ablations are wrong. Includes traces with 45+ question_ids.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "semantic_metrics_logs"
BENCH = ROOT / "strategyqa_results"

# Known benchmark outputs (same 50-Q slice: HF rows 40-89, question_id = 39 + question_num)
JSONL_RUNS: dict[str, Path] = {
    "both_095001": BENCH / "strategyqa_benchmark_20260414_095001.jsonl",
    "both_160235": BENCH / "strategyqa_benchmark_20260414_160235.jsonl",
    "neither_200850": BENCH / "strategyqa_benchmark_20260414_200850.jsonl",
    "density08_000224": BENCH / "strategyqa_benchmark_20260415_000224.jsonl",
    "density07_090706": BENCH / "strategyqa_benchmark_20260415_090706.jsonl",
    "density09_210431": BENCH / "strategyqa_benchmark_20260415_210431.jsonl",
    "entropy165_213252": BENCH / "strategyqa_benchmark_20260414_213252.jsonl",
    "entropy11_105654": BENCH / "strategyqa_benchmark_20260416_105654.jsonl",
}

# Trace files to scan (strategyqa); pair to logical run by prior manual alignment + timestamp
TRACE_TO_LABEL: list[tuple[str, Path]] = [
    ("both", LOGS / "trace_20260414_094345_2284928.jsonl"),
    ("both_b", LOGS / "trace_20260414_160229_3194264.jsonl"),
    ("neither", LOGS / "trace_20260414_200844_3218330.jsonl"),
    ("density08", LOGS / "trace_20260415_000218_3275869.jsonl"),
    ("density07", LOGS / "trace_20260415_090658_221385.jsonl"),
    ("density09", LOGS / "trace_20260415_210407_452897.jsonl"),
    ("entropy165", LOGS / "trace_20260414_213239_3275637.jsonl"),
    ("entropy11", LOGS / "trace_20260416_105653_911678.jsonl"),
]


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    idx = 0
    out: list[dict] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = dec.raw_decode(text, idx)
        out.append(obj)
        idx = end
    return out


def answer_from_ceo_text(s: str) -> str | None:
    if not s:
        return None
    m = re.search(r'"answer"\s*:\s*"(yes|no)"', s.lower())
    if m:
        return m.group(1)
    lo = s.lower()
    if "yes" in lo and "no" not in lo:
        return "yes"
    if "no" in lo and "yes" not in lo:
        return "no"
    return None


def scan_trace(path: Path) -> tuple[int, dict[str, dict]]:
    """Return (n_unique_qids, per_qid stats)."""
    per: dict[str, dict] = defaultdict(
        lambda: {
            "finishes": 0,
            "reprompt_flags": 0,
            "concern_true": 0,
            "multi_round": False,
            "ceo_final": None,
        }
    )
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("benchmark_name") != "strategyqa":
                continue
            qid = e.get("question_id")
            if qid is None:
                continue
            qid = str(qid)
            ev = e.get("event")
            if ev == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents"):
                per[qid]["finishes"] += 1
                if e.get("worker_reprompted_after_semantic_check"):
                    per[qid]["reprompt_flags"] += 1
                if e.get("semantic_quality_concern"):
                    per[qid]["concern_true"] += 1
            if ev == "ceo_final_answer":
                per[qid]["ceo_final"] = e.get("ceo_final_answer")
    for qid, d in per.items():
        d["multi_round"] = d["finishes"] > 1
        d["answer_guess"] = answer_from_ceo_text(str(d.get("ceo_final") or ""))
    return len(per), dict(per)


def main() -> None:
    # Load correctness by question_num (1..50) for each jsonl run
    gold_by_qn: dict[int, str] = {}
    correct_by_run: dict[str, dict[int, bool]] = {}
    qtext_by_qn: dict[int, str] = {}

    ref = load_jsonl(JSONL_RUNS["both_095001"])
    for r in ref:
        qn = int(r["question_num"])
        gold_by_qn[qn] = str(r["correct_answer"])
        qtext_by_qn[qn] = str(r.get("question", ""))[:120]

    for run, p in JSONL_RUNS.items():
        recs = load_jsonl(p)
        correct_by_run[run] = {int(r["question_num"]): bool(r.get("is_correct")) for r in recs}

    print("=== Traces with 45+ strategyqa question_ids ===\n")
    trace_stats: list[tuple[str, str, int, Path]] = []
    for label, p in TRACE_TO_LABEL:
        if not p.exists():
            continue
        nq, per = scan_trace(p)
        if nq >= 45:
            trace_stats.append((label, p.name, nq, p))
            print(f"{label:12} {p.name:42} n_qids={nq}")

    print("\n=== Hunt: both_095001 correct + (reprompt OR multi-round OR concern) + key others wrong ===\n")
    both_corr = correct_by_run["both_095001"]
    tr_path = LOGS / "trace_20260414_094345_2284928.jsonl"
    _, per_trace = scan_trace(tr_path)

    hits = []
    for qn in range(1, 51):
        qid = str(39 + qn)
        if qid not in per_trace:
            continue
        t = per_trace[qid]
        if not both_corr.get(qn):
            continue
        helped = t["reprompt_flags"] > 0 or t["multi_round"] or t["concern_true"] > 0
        if not helped:
            continue
        others_wrong = []
        for other in ("neither_200850", "density08_000224", "entropy165_213252", "both_160235"):
            if not correct_by_run[other].get(qn):
                others_wrong.append(other)
        if len(others_wrong) < 1:
            continue
        hits.append(
            {
                "qn": qn,
                "qid": qid,
                "q": qtext_by_qn[qn],
                "gold": gold_by_qn[qn],
                "finishes": t["finishes"],
                "reprompt": t["reprompt_flags"],
                "concern_events": t["concern_true"],
                "others_wrong": others_wrong,
            }
        )

    print(f"Found {len(hits)} candidates\n")
    for h in hits[:20]:
        print(json.dumps(h, indent=2))

    # Relax: both correct + multi_round + at least neither wrong
    print("\n=== Relax: both correct + finishes>=2 + neither wrong ===\n")
    hits2 = []
    for qn in range(1, 51):
        qid = str(39 + qn)
        if qid not in per_trace:
            continue
        t = per_trace[qid]
        if not both_corr.get(qn) or t["finishes"] < 2:
            continue
        if correct_by_run["neither_200850"].get(qn):
            continue
        hits2.append((qn, qid, t["finishes"], t["reprompt_flags"], t["concern_true"]))
    print(f"count={len(hits2)}", hits2[:15])

    # both correct, density08 wrong, trace shows concern or multi
    print("\n=== both correct + density08 wrong + (concern or finishes>=2) ===\n")
    hits3 = []
    for qn in range(1, 51):
        qid = str(39 + qn)
        if qid not in per_trace:
            continue
        t = per_trace[qid]
        if not both_corr.get(qn) or correct_by_run["density08_000224"].get(qn):
            continue
        if t["concern_true"] == 0 and t["finishes"] < 2:
            continue
        hits3.append((qn, qid, t["finishes"], t["concern_true"], t["reprompt_flags"]))
    for row in hits3:
        print(row, qtext_by_qn[row[0]][:80])

    # entropy165: correct with reprompt path while both wrong?
    print("\n=== entropy165 correct + both wrong + trace evidence (213239) ===\n")
    ep = LOGS / "trace_20260414_213239_3275637.jsonl"
    if ep.exists():
        _, per_e = scan_trace(ep)
        for qn in range(1, 51):
            if not correct_by_run["entropy165_213252"].get(qn):
                continue
            if correct_by_run["both_095001"].get(qn):
                continue
            qid = str(39 + qn)
            if qid not in per_e:
                continue
            te = per_e[qid]
            if te["finishes"] < 2 and te["concern_true"] == 0 and te["reprompt_flags"] == 0:
                continue
            print("qn", qn, "qid", qid, "fin", te["finishes"], "conc", te["concern_true"], "rep", te["reprompt_flags"])

    # Summary counts: when both correct, how often neither wrong
    n_both = sum(1 for qn in range(1, 51) if both_corr.get(qn))
    n_neither = sum(1 for qn in range(1, 51) if correct_by_run["neither_200850"].get(qn))
    agree = sum(
        1
        for qn in range(1, 51)
        if both_corr.get(qn) == correct_by_run["neither_200850"].get(qn)
    )
    print("\n=== Summary ===")
    print("both correct:", n_both, "/50")
    print("neither correct:", n_neither, "/50")
    print("same label both vs neither:", agree, "/50")


if __name__ == "__main__":
    main()
