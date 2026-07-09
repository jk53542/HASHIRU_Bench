#!/usr/bin/env python3
"""Generate play-by-play text from a StrategyQA orchestration trace (selected question_ids).

Trace ``question_id`` is the StrategyQA dataframe index (see ``benchmarking_strategyQA.py``),
not the 1-based ``question_num`` in the results file. Results are joined by exact ``question`` text.

Run::

    cd HASHIRU_Bench/bench/eval_results
    python _generate_strategyqa_playbyplay.py

Output: strategyqa_trace_8examples_playbyplay.txt
"""
from __future__ import annotations

import json
from pathlib import Path

TRACE = (
    Path(__file__).resolve().parent.parent
    / "results/semantic_metrics_logs/trace_20260406_003728_13082.jsonl"
)
RESULTS = (
    Path(__file__).resolve().parent.parent
    / "strategyqa_results/strategyqa_benchmark_20260406_003738.jsonl"
)
OUT = Path(__file__).resolve().parent / "strategyqa_trace_8examples_playbyplay.txt"

# Trace question_id = ChilleD/StrategyQA row index from benchmark ``iterrows()``.
SECTIONS: list[tuple[str, str, str]] = [
    (
        "286",
        "Category A — Single AskAgent; is_correct=True",
        "Boolean algebra described as binary",
    ),
    (
        "653",
        "Category B — AskMultipleAgents; is_correct=True",
        "Lumberjacks and three dosa",
    ),
    (
        "257",
        "Category C — Two AskAgent rounds; is_correct=False",
        "USAF discount at Dunkin Donuts",
    ),
    (
        "318",
        "Category D — AskMultipleAgents; is_correct=True",
        "Ancient Greece poleis vs US states in 1900",
    ),
    (
        "165",
        "Category E — AskMultipleAgents; is_correct=False",
        "Bloomberg fund Micronesia debt a decade",
    ),
    (
        "439",
        "Category F — AskMultipleAgents; is_correct=True",
        "Hunt Iberian wolves in Southern US",
    ),
    (
        "592",
        "Category G — Many worker rounds + AskMultipleAgents; is_correct=False",
        "Apollos vs D'Artagnans hypothetical fight",
    ),
    (
        "654",
        "Category H — AskMultipleAgents; is_correct=False",
        "Wembley crowd and Mongol descendants",
    ),
]


def _norm_q(s: str) -> str:
    return " ".join((s or "").split())


def load_strategyqa_results(path: Path) -> dict[str, dict]:
    """Map normalized question text -> result row."""
    raw = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    idx = 0
    out: dict[str, dict] = {}
    L = len(raw)
    while idx < L:
        while idx < L and raw[idx].isspace():
            idx += 1
        if idx >= L:
            break
        obj, end = dec.raw_decode(raw, idx)
        idx = end
        q = obj.get("question")
        if isinstance(q, str) and q.strip():
            out[_norm_q(q)] = obj
    return out


def load_events_for_question(trace_path: Path, qid: str) -> tuple[str, list[dict]]:
    question_text = ""
    events: list[dict] = []
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(o.get("question_id", "")) != str(qid):
                continue
            events.append(o)
            if not question_text:
                qt = o.get("question_text")
                if isinstance(qt, str) and qt.strip():
                    question_text = qt.strip()
    events.sort(key=lambda x: float(x.get("ts") or 0.0))
    return question_text, events


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
    return "; ".join(parts) if parts else "(no top-level metrics)"


def emit_round(buf: list[str], idx: int, e: dict) -> None:
    tool = e.get("tool")
    buf.append(f"\n--- Worker round {idx + 1} (ceo_tool_finished, tool={tool}) ---\n")
    buf.append(f"Tool status/message: {e.get('status')!r} / {e.get('message')!r}\n")
    buf.append(f"Metrics: {fmt_metrics(e)}\n")
    if tool == "AskAgent":
        buf.append(f"Agent: {e.get('agent_name')!r}\n")
        buf.append("\nPROMPT (exact):\n")
        buf.append(str(e.get("worker_prompt") or e.get("args", {}).get("prompt") or ""))
        buf.append("\n\nRESPONSE (exact):\n")
        buf.append(str(e.get("worker_response") or ""))
        buf.append("\n")
    elif tool == "AskMultipleAgents":
        pa = e.get("per_agent_outputs")
        uq = e.get("user_question") or (e.get("args") or {}).get("user_question")
        if uq:
            buf.append(f"\nUser question passed to tool (exact):\n{uq}\n")
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
    buf.append("\n")


def playbyplay_for(
    qid: str,
    title_note: str,
    short_label: str,
    by_question: dict[str, dict],
) -> str:
    qtext, events = load_events_for_question(TRACE, qid)
    finishes = [
        e
        for e in events
        if e.get("event") == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    n = len(finishes)
    reprompts = max(0, n - 1)

    res = by_question.get(_norm_q(qtext), {})

    lines: list[str] = []
    lines.append("=" * 88 + "\n")
    lines.append(
        f"QUESTION ID: {qid} (StrategyQA dataset row index in trace; "
        f"not the benchmark run question_num)\n"
    )
    lines.append(f"Section label: {title_note}\n")
    lines.append(f"Topic (short): {short_label}\n")
    if res:
        lines.append(
            f"Benchmark (results JSONL): question_num={res.get('question_num')!r}, "
            f"is_correct={res.get('is_correct')!r}, "
            f"correct_answer={res.get('correct_answer')!r}, "
            f"agent_resp={res.get('agent_resp')!r}\n"
        )
    else:
        lines.append("(No matching row in results file for this question_text.)\n")
    lines.append(f"Worker tool rounds (AskAgent / AskMultipleAgents): {n}\n")
    lines.append(f"CEO reprompts after first worker round: {reprompts}\n")
    lines.append("\n--- ORIGINAL BENCHMARK QUESTION (question_text) ---\n")
    lines.append(qtext or "(missing in trace)\n")
    lines.append("\n")

    lines.append("--- HIGH-LEVEL SUMMARY ---\n")
    if n == 0:
        lines.append("No AskAgent/AskMultipleAgents completions found for this question_id.\n")
        return "".join(lines)
    lines.append(
        f"The CEO invoked worker tools {n} time(s) for this item. "
        f"That is 1 initial delegation plus {reprompts} subsequent worker cycle(s).\n"
    )
    tools_seq = [e.get("tool") for e in finishes]
    lines.append(f"Sequence of tools: {' → '.join(str(t) for t in tools_seq)}\n")
    thresh_flags = sum(
        1
        for e in finishes
        if e.get("semantic_quality_concern")
        or ("crossed thresholds" in str(e.get("message") or "").lower())
    )
    if thresh_flags:
        lines.append(
            f"Threshold-style tool messages or semantic_quality_concern on {thresh_flags} finish row(s).\n"
        )
    lines.append("\n--- FULL PLAY-BY-PLAY (exact prompts and responses) ---\n")
    for i, e in enumerate(finishes):
        emit_round(lines, i, e)

    return "".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    by_q = load_strategyqa_results(RESULTS) if RESULTS.exists() else {}

    header: list[str] = []
    header.append("StrategyQA trace play-by-play\n")
    header.append(f"Trace: {TRACE.name}\n")
    header.append(f"Results: {RESULTS.name}\n")
    header.append(
        "Trace question_id is the StrategyQA dataframe index passed to hashiru_trace_context_prefix "
        "(see benchmarking_strategyQA.py). Join to results by question string.\n\n"
        "--- MASTER SUMMARY (eight categories A–H) ---\n"
        "A 286: One AskAgent; LogicExpert; correct.\n"
        "B 653: AskMultipleAgents; correct (lumberjack calories vs dosa).\n"
        "C 257: Two AskAgent rounds; wrong (military discount).\n"
        "D 318: AskMultipleAgents; correct (Greece city-states vs US states).\n"
        "E 165: AskMultipleAgents; wrong (Bloomberg vs Micronesia budget).\n"
        "F 439: AskMultipleAgents; correct (Iberian wolves / geography).\n"
        "G 592: Long chain including AskMultipleAgents; wrong (Apollo vs D'Artagnan).\n"
        "H 654: AskMultipleAgents; wrong (Mongol descendants at Wembley).\n\n"
        "Each section lists question_text, then every ceo_tool_finished worker cycle.\n\n"
    )

    body_parts = ["".join(header)]
    for qid, cat, label in SECTIONS:
        body_parts.append(playbyplay_for(qid, cat, label, by_q))
        body_parts.append("\n\n")

    OUT.write_text("".join(body_parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
