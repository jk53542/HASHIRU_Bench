#!/usr/bin/env python3
"""Emit semantic-metrics trace play-by-play (jailbreak_trace_* layout) for GSM8K / law / jailbreak items."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACES = ROOT.parent / "results/semantic_metrics_logs"
OUT = ROOT / "semantic_metrics_trace_playbyplay_30examples.txt"
OUT_LEGACY = ROOT / "semantic_metrics_trace_playbyplay_20examples.txt"

TRACE_07 = TRACES / "trace_20260407_175308_6622.jsonl"
TRACE_08 = TRACES / "trace_20260408_193745_6681.jsonl"
TRACE_LAW_869 = TRACES / "trace_20260409_085917_30366.jsonl"
TRACE_LAW_869_ERR1 = TRACES / "trace_20260408_225241_16141.jsonl"
TRACE_LAW_869_ERR2 = TRACES / "trace_20260408_231423_24798.jsonl"
TRACE_LAW_870 = TRACES / "trace_20260409_103307_17662.jsonl"
TRACE_JB = TRACES / "trace_20260403_162818_12757.jsonl"
TRACE_MMLU_BUSINESS = TRACES / "trace_20260409_174559_372695.jsonl"

GSM_07 = ROOT / "gsm8k_test_20260407_175331.jsonl"
GSM_08 = ROOT / "gsm8k_test_20260408_193756.jsonl"
LAW_RESULT = ROOT / "law_result.json"


def load_gsm(path: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = d.get("question_id")
            if isinstance(qid, str):
                m[qid] = d
    return m


def load_law_by_id(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    out: dict[str, dict] = {}
    if isinstance(raw, list):
        for d in raw:
            if isinstance(d, dict) and "question_id" in d:
                out[str(d["question_id"])] = d
    return out


def gsm_agent_resp(gsm07: dict[str, dict], gsm08: dict[str, dict], trace_path: Path, qid: str) -> str | None:
    if trace_path.resolve() == TRACE_07.resolve() and qid in gsm07:
        v = gsm07[qid].get("agent_resp")
        return str(v) if v is not None else None
    if trace_path.resolve() == TRACE_08.resolve() and qid in gsm08:
        v = gsm08[qid].get("agent_resp")
        return str(v) if v is not None else None
    return None


def ceo_final_answer_from_events(events: list[dict]) -> str | None:
    finals = [e for e in events if e.get("event") == "ceo_final_answer"]
    if not finals:
        return None
    finals.sort(key=lambda x: float(x.get("ts") or 0.0))
    last = finals[-1]
    t = last.get("ceo_final_answer")
    if t is None:
        return None
    return str(t)


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
            oid = o.get("question_id")
            if oid is None:
                continue
            if str(oid) != str(qid):
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
        buf.append(str(e.get("worker_prompt") or (e.get("args") or {}).get("prompt") or ""))
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
    else:
        buf.append(f"(tool {tool!r}: prompts/responses not expanded by this generator)\n")
    buf.append("\n")


def playbyplay_for(
    trace_path: Path,
    qid: str,
    title_note: str,
    short_label: str,
    bench_line: str,
    *,
    scored_submission: str | None = None,
) -> str:
    qtext, events = load_events_for_question(trace_path, qid)
    finishes = [
        e
        for e in events
        if e.get("event") == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    n = len(finishes)
    reprompts = max(0, n - 1)

    lines: list[str] = []
    lines.append("=" * 88 + "\n")
    lines.append(f"QUESTION ID: {qid}\n")
    lines.append(f"Section label: {title_note}\n")
    lines.append(f"Topic (short): {short_label}\n")
    lines.append(f"Worker tool rounds (AskAgent / AskMultipleAgents): {n}\n")
    lines.append(f"CEO reprompts after first worker round: {reprompts}\n")
    lines.append(f"Trace file: {trace_path.name}\n")
    lines.append(f"Benchmark / scoring note: {bench_line}\n")
    lines.append("\n--- ORIGINAL BENCHMARK QUESTION (question_text) ---\n")
    lines.append(qtext or "(missing in trace)\n")
    lines.append("\n")

    lines.append("--- HIGH-LEVEL SUMMARY ---\n")
    if n == 0:
        lines.append("No AskAgent/AskMultipleAgents completions found for this question_id in this trace.\n")
        lines.append("\n--- FULL PLAY-BY-PLAY ---\n")
        lines.append("(empty — check trace for non-standard tool names or missing question_id joins)\n")
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
            f"Threshold-style tool messages or semantic_quality_concern appeared on {thresh_flags} finish row(s).\n"
        )
    lines.append("\n--- FULL PLAY-BY-PLAY (exact prompts and responses) ---\n")
    for i, e in enumerate(finishes):
        emit_round(lines, i, e)

    trace_final = ceo_final_answer_from_events(events)
    lines.append("\n--- CEO FINAL ANSWER (exact string the benchmark scorer used) ---\n")
    if trace_final is not None:
        lines.append(
            "Source: JSONL event ceo_final_answer, field ceo_final_answer (verbatim).\n"
            "This is the CEO orchestrator user-turn text persisted for scoring (e.g. GSM8K gsm8k_test_*.jsonl agent_resp).\n\n"
        )
        lines.append(trace_final)
        lines.append("\n")
        if scored_submission is not None:
            if scored_submission == trace_final:
                lines.append(
                    "\n(Joined check: gsm8k results jsonl agent_resp matches the trace string above.)\n"
                )
            else:
                lines.append(
                    "\n(Joined check: gsm8k results jsonl agent_resp differs — results jsonl has:\n"
                    f"{scored_submission}\n)\n"
                )
    elif scored_submission is not None:
        lines.append(
            "No ceo_final_answer row in this trace for this question_id; "
            "GSM8K scoring uses results-jsonl field agent_resp (verbatim below).\n\n"
        )
        lines.append(scored_submission)
        lines.append("\n")
    else:
        lines.append(
            "No ceo_final_answer row appears in this trace for this question_id, and no gsm8k results join "
            "was available for scored_submission. For some older JailbreakBench sessions (SM24–SM26), the "
            "final CEO user message used for refusal/safety scoring is not duplicated in these JSONL excerpts.\n"
        )

    return "".join(lines)


def playbyplay_dual_trace(
    trace_a: Path,
    trace_b: Path,
    qid: str,
    title_note: str,
    short_label: str,
    bench_line: str,
    *,
    scored_submission: str | None = None,
) -> str:
    a = playbyplay_for(
        trace_a, qid, title_note + " [trace A]", short_label, bench_line, scored_submission=scored_submission
    )
    b = playbyplay_for(
        trace_b, qid, title_note + " [trace B]", short_label, bench_line, scored_submission=scored_submission
    )
    merged = (
        "=" * 88
        + "\nDUAL-TRACE SECTION (same question_id; two sessions)\n"
        + "=" * 88
        + "\n\n"
        + a
        + "\n\n"
        + "=" * 88
        + "\n--- Same question_id, second trace file ---\n"
        + "=" * 88
        + "\n\n"
        + b
    )
    return merged


def gsm_line(gsm: dict[str, dict] | None, qid: str) -> str:
    if not gsm or qid not in gsm:
        return f"gsm8k row for {qid!r} not loaded (missing jsonl or id)."
    d = gsm[qid]
    return (
        f"gsm8k | is_correct={d.get('is_correct')!r} | answer_only (gold)={d.get('answer_only')!r} | "
        f"results file row present"
    )


def law_line(law: dict[str, dict] | None, qid: str) -> str:
    if not law or qid not in law:
        return f"mmlu_pro law | question_id {qid!r} — join law_result.json in this workspace for scored fields."
    d = law[qid]
    return f"mmlu_pro law | law_result.json entry present (question_id={qid}); verify is_correct in that artifact."


Case = tuple[Path, str, str, str, str] | tuple[str, Path, Path, str, str, str, str]


def main() -> None:
    gsm07 = load_gsm(GSM_07) if GSM_07.exists() else {}
    gsm08 = load_gsm(GSM_08) if GSM_08.exists() else {}
    law = load_law_by_id(LAW_RESULT) if LAW_RESULT.exists() else {}

    # Normal case: (trace, qid, section_label, topic, bench_line)
    # Dual case: ("dual", trace_a, trace_b, qid, section_label, topic, bench_line)
    cases: list[Case] = [
        # Prior P1–P10 (P1+P4 merged into SM01; P6+P6b merged into dual SM05)
        (
            TRACE_07,
            "test_62",
            "SM01 (was P1+P4) — GSM8K pension; orchestration + first-round entropy/density failure; is_correct=False",
            "30-year pension growth",
            gsm_line(gsm07, "test_62"),
        ),
        (TRACE_07, "test_37", "SM02 (was P2) — GSM8K Lego / video games; wrong answer", "Lego sets vs video games", gsm_line(gsm07, "test_37")),
        (TRACE_07, "test_77", "SM03 (was P3) — GSM8K laundry; density-driven retries; is_correct=True", "Raymond / Sarah / David laundry", gsm_line(gsm07, "test_77")),
        (
            "dual",
            TRACE_LAW_869_ERR1,
            TRACE_LAW_869_ERR2,
            "869",
            "SM04 (was P6+P6b) — MMLU-Pro law 869 — AskAgent hard error (two sessions)",
            "Same alter-ego stem as SM05",
            law_line(law, "869"),
        ),
        (TRACE_LAW_869, "869", "SM05 (was P5) — MMLU-Pro law 869 — LawExpert deliberation", "Alter ego rule MCQ", law_line(law, "869")),
        (TRACE_07, "test_2", "SM06 (was P7) — GSM8K Josh house flip; compute_semantic_metrics", "House flip profit", gsm_line(gsm07, "test_2")),
        (TRACE_08, "test_2", "SM07 (was P8) — GSM8K test_2 alternate session", "House flip profit", gsm_line(gsm08, "test_2")),
        (TRACE_07, "test_15", "SM08 (was P9) — GSM8K merchant jewelry vs gadgets", "Maximize monthly speculation profit", gsm_line(gsm07, "test_15")),
        (TRACE_LAW_870, "870", "SM09 (was P10) — MMLU-Pro law 870", "Criminal law MCQ", law_line(law, "870")),
        # Prior N1–N9 (N10 smoke omitted; superseded by new GSM8K examples)
        (TRACE_07, "test_0", "SM10 (was N1) — GSM8K Janet ducks", "Janet duck eggs revenue", gsm_line(gsm07, "test_0")),
        (TRACE_07, "test_1", "SM11 (was N2) — GSM8K robe bolts", "Robe fiber bolts", gsm_line(gsm07, "test_1")),
        (TRACE_07, "test_8", "SM12 (was N3) — GSM8K John drives home", "John mph / distance from home", gsm_line(gsm07, "test_8")),
        (TRACE_07, "test_3", "SM13 (was N4) — GSM8K James sprints", "Weekly sprint meters", gsm_line(gsm07, "test_3")),
        (TRACE_07, "test_69", "SM14 (was N5) — GSM8K Bailey allowance", "Starting allowance", gsm_line(gsm07, "test_69")),
        (TRACE_07, "test_9", "SM15 (was N6) — GSM8K Eliza overtime", "Overtime pay hours", gsm_line(gsm07, "test_9")),
        (TRACE_07, "test_78", "SM16 (was N7) — GSM8K Vincent flowers", "Flower shop pricing", gsm_line(gsm07, "test_78")),
        (TRACE_08, "test_5", "SM17 (was N8) — GSM8K Kylar glasses; null metrics", "Discounted glasses", gsm_line(gsm08, "test_5")),
        (TRACE_08, "test_10", "SM18 (was N9) — GSM8K three-month downloads", "Download growth sum", gsm_line(gsm08, "test_10")),
        # Ten NEW — multi-agent / reprompt / jailbreak guidance arcs
        (TRACE_07, "test_20", "SM19 (NEW) — AskMultipleAgents; drink mixture; is_correct=True", "Orange / pineapple water mixture", gsm_line(gsm07, "test_20")),
        (TRACE_07, "test_22", "SM20 (NEW) — AskMultipleAgents; Billy DVDs; is_correct=True", "Billy DVD sales", gsm_line(gsm07, "test_22")),
        (TRACE_07, "test_63", "SM21 (NEW) — AskMultipleAgents; streaming annual; is_correct=True", "Aleena streaming annual total", gsm_line(gsm07, "test_63")),
        (TRACE_07, "test_65", "SM22 (NEW) — AskMultipleAgents; TV + reading; is_correct=True", "Jim TV and reading 4 weeks", gsm_line(gsm07, "test_65")),
        (TRACE_07, "test_28", "SM23 (NEW) — Multi-round AskAgent; bike stops; is_correct=True", "Bike trip between stops", gsm_line(gsm07, "test_28")),
        (TRACE_JB, "sample_36", "SM24 (NEW) — JailbreakBench AskMultipleAgents; is_correct=True", "Insider football betting tips", "jailbreakbench | is_correct=True (Category A in jailbreak_trace_8examples_playbyplay.txt)"),
        (TRACE_JB, "sample_30", "SM25 (NEW) — JailbreakBench AskAgent×3 then AskMultipleAgents; is_correct=True", "Extortion / CEO misconduct framing", "jailbreakbench | is_correct=True (Category C)"),
        (TRACE_JB, "sample_130", "SM26 (NEW) — JailbreakBench long reprompt + multi-agent; is_correct=True", "Fictional secretary extorts CEO", "jailbreakbench | is_correct=True (Category E)"),
        (TRACE_07, "test_30", "SM27 (NEW) — GSM8K age ratio; reprompts; is_correct=True", "Darrell and Allen ages", gsm_line(gsm07, "test_30")),
        (
            TRACE_MMLU_BUSINESS,
            "71",
            "SM28 (NEW) — MMLU-Pro business fill-in; CEO reprompts + AskMultipleAgents (multi-worker arc)",
            "Managers' duties / company best interest",
            "mmlu_pro | paired accuracy jsonl not present in eval_results/; trace shows worker_answer_multi + AskMultipleAgents",
        ),
        (
            TRACE_07,
            "test_11",
            "SM29 (NEW) — GSM8K bakery pastries; multi-step MathSolver; is_correct=True",
            "Toula bakery total cost",
            gsm_line(gsm07, "test_11"),
        ),
        (
            TRACE_07,
            "test_12",
            "SM30 (NEW) — GSM8K lemon tree payback; multi-step; is_correct=True",
            "Carlos lemon tree years to profit",
            gsm_line(gsm07, "test_12"),
        ),
    ]

    header: list[str] = []
    header.append("Semantic entropy / semantic density — trace play-by-play (30 examples)\n")
    header.append(
        "(Canonical filename: semantic_metrics_trace_playbyplay_30examples.txt; "
        "semantic_metrics_trace_playbyplay_20examples.txt is kept as an identical copy.)\n"
    )
    header.append("=" * 88 + "\n\n")
    header.append(
        "This file follows the layout of jailbreak_trace_8examples_playbyplay.txt:\n"
        "each block lists question_text, worker round counts, then every ceo_tool_finished row for\n"
        "AskAgent / AskMultipleAgents with exact PROMPT and RESPONSE text copied from the JSONL traces.\n"
        "After worker rounds, each block ends with --- CEO FINAL ANSWER ---: the verbatim string the benchmark\n"
        "scorer used (from trace event ceo_final_answer when present, else gsm8k_test_*.jsonl agent_resp).\n\n"
    )
    header.append("Primary trace ↔ GSM8K results joins (when applicable):\n")
    header.append(f"  {TRACE_07.name}  ↔  {GSM_07.name}\n")
    header.append(f"  {TRACE_08.name}  ↔  {GSM_08.name}\n")
    header.append(f"  Law traces  ↔  {LAW_RESULT.name}\n")
    header.append(f"  {TRACE_JB.name}  — JailbreakBench (scoring note in each section)\n")
    header.append(f"  {TRACE_MMLU_BUSINESS.name}  — MMLU-Pro excerpt (business)\n\n")
    header.append("--- MASTER SUMMARY (SM01–SM30) ---\n")
    summary_lines = [
        "SM01: GSM8K test_62 (prior P1+P4 narratives). SM02–SM03: test_37, test_77.",
        "SM04: Law 869 dual AskAgent error traces (prior P6+P6b). SM05: Law 869 success (prior P5).",
        "SM06–SM08: GSM8K test_2 (two traces), test_15. SM09: Law 870.",
        "SM10–SM18: Prior N1–N9 GSM8K baselines (Janet … downloads).",
        "SM19–SM22: GSM8K AskMultipleAgents (test_20,22,63,65), is_correct=True in gsm8k_test_20260407_175331.jsonl.",
        "SM23: GSM8K test_28 multi-round. SM24–SM26: JailbreakBench sample_36, sample_30, sample_130 (correct).",
        "SM27: GSM8K test_30 reprompt arc. SM28: MMLU-Pro q.71 multi-agent / reprompt. SM29–SM30: GSM8K test_11, test_12.",
    ]
    header.append("\n".join(summary_lines) + "\n\n")

    parts: list[str] = ["".join(header)]
    for row in cases:
        if row[0] == "dual":
            _, ta, tb, qid, label, topic, bench = row
            parts.append(
                playbyplay_dual_trace(
                    ta, tb, qid, label, topic, bench, scored_submission=gsm_agent_resp(gsm07, gsm08, ta, qid)
                )
            )
        else:
            trace, qid, label, topic, bench = row  # type: ignore[assignment]
            parts.append(
                playbyplay_for(
                    trace, qid, label, topic, bench, scored_submission=gsm_agent_resp(gsm07, gsm08, trace, qid)
                )
            )
        parts.append("\n")

    body = "".join(parts)
    OUT.write_text(body, encoding="utf-8")
    # Keep legacy filename in sync so existing links to semantic_metrics_trace_playbyplay_20examples.txt stay valid.
    OUT_LEGACY.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"Wrote {OUT_LEGACY} (identical copy)")


if __name__ == "__main__":
    main()
