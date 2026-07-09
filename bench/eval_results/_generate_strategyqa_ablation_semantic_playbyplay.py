#!/usr/bin/env python3
"""
Emit strategyqa_ablation_semantic_helped_playbyplay.txt — StrategyQA ablation
verbatim play-by-play (same layout spirit as semantic_metrics_trace_playbyplay_30examples.txt).

For GSM8K / TruthfulQA / MMLU law / JailbreakBench / IFBench / ToolBench / τ² index, see
_generate_benchmark_ablation_playbyplays.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "semantic_metrics_logs"
BENCH = ROOT / "strategyqa_results"
OUT = Path(__file__).resolve().parent / "strategyqa_ablation_semantic_helped_playbyplay.txt"

JSONL = {
    "both": BENCH / "strategyqa_benchmark_20260414_095001.jsonl",
    "neither": BENCH / "strategyqa_benchmark_20260414_200850.jsonl",
    "density08": BENCH / "strategyqa_benchmark_20260415_000224.jsonl",
    "entropy165": BENCH / "strategyqa_benchmark_20260414_213252.jsonl",
}

TRACES = {
    "both (entropy+density, thr 1.65 / 0.8)": LOGS / "trace_20260414_094345_2284928.jsonl",
    "neither (metrics off)": LOGS / "trace_20260414_200844_3218330.jsonl",
    "density only (thr 0.8)": LOGS / "trace_20260415_000218_3275869.jsonl",
    "entropy only @1.65 (density off)": LOGS / "trace_20260414_213239_3275637.jsonl",
}

CASES = [
    {
        "sid": "SQ01",
        "qid": "42",
        "qn": 3,
        "title": "Jason vs Dr. Disrespect — common ground?",
        "note": "Both-metrics trace shows concern+reprompt; both JSONL correct; neither+entropy165 JSONL wrong.",
    },
    {
        "sid": "SQ02",
        "qid": "50",
        "qn": 11,
        "title": "Jon Brower Minnoch — anorexia nervosa?",
        "note": "Both correct after reprompt-marked round; neither JSONL wrong.",
    },
    {
        "sid": "SQ03",
        "qid": "74",
        "qn": 35,
        "title": "Pitt founder vs Judith Sheindlin — much in common?",
        "note": "Both correct after long multi-round arc; neither+density08+entropy165 JSONL wrong.",
    },
    {
        "sid": "SQ04",
        "qid": "53",
        "qn": 14,
        "title": "Letter B vs Prince Harry birth order",
        "note": "Density-only JSONL correct with concern+reprompt; both+neither JSONL wrong.",
    },
    {
        "sid": "SQ05",
        "qid": "63",
        "qn": 24,
        "title": "Jockey / Triple Crown / Eid al-Fitr window",
        "note": "Density-only correct after many low-density rounds; both+neither wrong.",
    },
    {
        "sid": "SQ06",
        "qid": "75",
        "qn": 36,
        "title": "Samsung Galaxy 1 OS — sound edible?",
        "note": "Bonus: density-only correct with reprompt; both+neither wrong.",
    },
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


def row_for_qn(recs: list[dict], qn: int) -> dict | None:
    for r in recs:
        if int(r.get("question_num", -1)) == qn:
            return r
    return None


def fmt_metrics(e: dict) -> str:
    parts = []
    for k in (
        "semantic_entropy",
        "semantic_density",
        "semantic_entropy_threshold",
        "semantic_density_threshold",
        "semantic_quality_concern",
        "worker_reprompted_after_semantic_check",
    ):
        if k in e:
            parts.append(f"{k}={e.get(k)!r}")
    return "; ".join(parts) if parts else "(no top-level metrics on finish row)"


def emit_finish(buf: list[str], idx: int, e: dict) -> None:
    tool = e.get("tool")
    buf.append(f"\n--- Worker round {idx + 1} (ceo_tool_finished, tool={tool}) ---\n")
    buf.append(f"Tool status/message: {e.get('status')!r} / {e.get('message')!r}\n")
    buf.append(f"Metrics: {fmt_metrics(e)}\n")

    if tool == "AskAgent":
        buf.append(f"Agent: {e.get('agent_name')!r}\n")
        buf.append("\nPROMPT (exact):\n")
        buf.append(str(e.get("worker_prompt") or (e.get("args") or {}).get("prompt") or ""))
        buf.append("\n\nRESPONSE (exact):\n")
        buf.append(str(e.get("worker_response") or ""))
        buf.append("\n")
        return

    if tool == "AskMultipleAgents":
        uq = e.get("user_question") or (e.get("args") or {}).get("user_question")
        if uq:
            buf.append(f"\nUser question passed to tool (exact):\n{uq}\n")
        pa = e.get("per_agent_outputs")
        if isinstance(pa, list):
            for j, row in enumerate(pa):
                if not isinstance(row, dict):
                    continue
                buf.append(
                    f"\n--- Sub-agent {j + 1}: {row.get('agent_name')!r} "
                    f"(base_model={row.get('base_model')!r}) ---\n"
                )
                buf.append("\nPROMPT (exact):\n")
                buf.append(str(row.get("prompt") or ""))
                buf.append("\n\nRESPONSE (exact):\n")
                buf.append(str(row.get("response") or ""))
                buf.append("\n")
                if "semantic_quality_concern" in row:
                    buf.append(
                        f"(per-agent: semantic_quality_concern={row.get('semantic_quality_concern')!r}, "
                        f"entropy={row.get('semantic_entropy')!r}, density={row.get('semantic_density')!r})\n"
                    )
        else:
            buf.append("(no per_agent_outputs on finish row)\n")
        return

    buf.append(f"(tool {tool!r}: prompts/responses not expanded)\n")


def events_for_qid(trace_path: Path, qid: str) -> list[dict]:
    out: list[dict] = []
    if not trace_path.exists():
        return out
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("benchmark_name") != "strategyqa":
                continue
            if str(e.get("question_id")) != qid:
                continue
            out.append(e)
    out.sort(key=lambda x: float(x.get("ts") or 0.0))
    return out


def build_ablation_section(
    label: str,
    trace_path: Path,
    qid: str,
    qn: int,
    bench_rows: dict[str, dict | None],
) -> str:
    buf: list[str] = []
    buf.append("\n" + "-" * 88 + "\n")
    buf.append(f"ABLATION: {label}\n")
    buf.append(f"Trace file: {trace_path.name}\n")
    ev = events_for_qid(trace_path, qid)
    if not ev:
        buf.append("(No strategyqa events for this question_id in this trace file.)\n")
        return "".join(buf)

    finishes = [
        e
        for e in ev
        if e.get("event") == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    finals = [e for e in ev if e.get("event") == "ceo_final_answer"]

    buf.append(f"Worker tool rounds (ceo_tool_finished): {len(finishes)}\n")
    rep = sum(1 for e in finishes if e.get("worker_reprompted_after_semantic_check") is True)
    buf.append(f"Rounds with worker_reprompted_after_semantic_check=True: {rep}\n")
    buf.append(f"Rounds with semantic_quality_concern=True: {sum(1 for e in finishes if e.get('semantic_quality_concern'))}\n")

    for name, rec in bench_rows.items():
        buf.append(
            f"JSONL ({name}): "
            + (
                f"is_correct={rec.get('is_correct')!r} gold={rec.get('correct_answer')!r} "
                f"agent_resp={rec.get('agent_resp')!r}"
                if rec
                else "(no row)"
            )
            + "\n"
        )

    buf.append("\n--- FULL PLAY-BY-PLAY (verbatim from trace) ---\n")
    if not finishes:
        buf.append("(No AskAgent / AskMultipleAgents ceo_tool_finished rows for this question_id.)\n")
    for i, e in enumerate(finishes):
        emit_finish(buf, i, e)

    buf.append("\n--- CEO FINAL ANSWER (verbatim from trace event ceo_final_answer) ---\n")
    if finals:
        buf.append(str(finals[-1].get("ceo_final_answer") or ""))
        buf.append("\n")
    else:
        buf.append("(no ceo_final_answer event for this question_id)\n")

    return "".join(buf)


def main() -> None:
    cached: dict[str, list[dict]] = {k: load_jsonl(p) for k, p in JSONL.items()}

    parts: list[str] = []
    parts.append(
        "StrategyQA semantic ablations — play-by-play (verbatim traces + JSONL scores)\n"
        "(Canonical filename: strategyqa_ablation_semantic_helped_playbyplay.txt)\n"
        "================================================================================\n\n"
        "This file mirrors the style of semantic_metrics_trace_playbyplay_30examples.txt:\n"
        "each case starts with question metadata, a high-level takeaway, then per-ablation\n"
        "sections with every ceo_tool_finished row for AskAgent / AskMultipleAgents (exact\n"
        "prompts and worker outputs from the orchestration JSONL), ending with the verbatim\n"
        "ceo_final_answer string from the trace.\n\n"
        "Trace ↔ benchmark JSONL references:\n"
        "  trace_20260414_094345_2284928.jsonl  ↔  strategyqa_benchmark_20260414_095001.jsonl (both on)\n"
        "  trace_20260414_200844_3218330.jsonl  ↔  strategyqa_benchmark_20260414_200850.jsonl (neither)\n"
        "  trace_20260415_000218_3275869.jsonl  ↔  strategyqa_benchmark_20260415_000224.jsonl (density only, thr=0.8)\n"
        "  trace_20260414_213239_3275637.jsonl  ↔  strategyqa_benchmark_20260414_213252.jsonl (entropy only, thr=1.65)\n\n"
        "Note: some trace files omit a few question_id values (known logging gaps); if a section\n"
        "says no events, rely on the JSONL row for scoring and treat the trace as incomplete.\n\n"
        "--- OVERALL TAKEAWAY (this document’s cases) ---\n"
        "On the shared 50-question StrategyQA slice (HF row indices 40–89; trace question_id = 39 + question_num),\n"
        "semantic metrics sometimes drive useful extra worker rounds (semantic_quality_concern and/or\n"
        "worker_reprompted_after_semantic_check) and correlate with correct final JSON in the benchmark\n"
        "JSONL — but the effect is item-specific and condition-specific. The “both” run (095001) scores\n"
        "highest overall accuracy on this slice partly because it combines entropy and density signals,\n"
        "while density-only fixes several “both” failures (e.g. SQ04–SQ06) yet loses others (e.g. SQ03)\n"
        "where aggressive low-density chasing pushes the CEO to the wrong yes/no.\n"
    )

    for case in CASES:
        qid = case["qid"]
        qn = case["qn"]
        ev0 = events_for_qid(TRACES["both (entropy+density, thr 1.65 / 0.8)"], qid)
        qtext = next(
            (e.get("question_text") for e in ev0 if isinstance(e.get("question_text"), str) and e.get("question_text")),
            "(missing question_text)",
        )

        bench_rows = {k: row_for_qn(cached[k], qn) for k in cached}

        parts.append("\n" + "=" * 92 + "\n")
        parts.append(f"{case['sid']}: question_id={qid} | question_num={qn}\n")
        parts.append(f"Title: {case['title']}\n")
        parts.append(f"Analyst note: {case['note']}\n\n")
        parts.append("--- ORIGINAL BENCHMARK QUESTION (question_text from trace) ---\n")
        parts.append(str(qtext) + "\n")
        parts.append("\n--- HIGH-LEVEL PLAY-BY-PLAY ---\n")
        finishes_b = [
            e
            for e in ev0
            if e.get("event") == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        seq = " → ".join(str(e.get("tool")) for e in finishes_b) or "(none)"
        parts.append(
            f"Both-metrics trace shows {len(finishes_b)} worker-tool completion(s). Tool sequence: {seq}.\n"
            f"Concern-marked finish rows: {sum(1 for e in finishes_b if e.get('semantic_quality_concern'))}. "
            f"Reprompt-marked rows: {sum(1 for e in finishes_b if e.get('worker_reprompted_after_semantic_check'))}.\n"
        )

        for label, tpath in TRACES.items():
            parts.append(build_ablation_section(label, tpath, qid, qn, bench_rows))

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
