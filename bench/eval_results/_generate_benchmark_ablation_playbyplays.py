#!/usr/bin/env python3
"""
Regenerate ablation-style play-by-play .txt files (same spirit as
strategyqa_ablation_semantic_helped_playbyplay.txt) for benchmarks that have
orchestration traces under bench/results/semantic_metrics_logs/.

Also writes `mmlu_pro_ablation_869_playbyplay.txt` (single high-churn MMLU-Pro law case) and
`ifbench_ablation_sd_se_vs_off_playbyplay.txt` (IFBench prompt-matched SD+SE vs off when both traces exist).

Run from repo:  python3 HASHIRU_Bench/bench/eval_results/_generate_benchmark_ablation_playbyplays.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "semantic_metrics_logs"
EVAL = Path(__file__).resolve().parent


def normalize_text(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip())


def load_concat_json_values(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    idx = 0
    out: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = dec.raw_decode(text, idx)
        if not isinstance(obj, dict):
            idx = end
            continue
        out.append(obj)
        idx = end
    return out


def events_for_qid(
    trace_path: Path, benchmark_name: str, question_id: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
            if e.get("benchmark_name") != benchmark_name:
                continue
            if str(e.get("question_id")) != str(question_id):
                continue
            out.append(e)
    out.sort(key=lambda x: float(x.get("ts") or 0.0))
    return out


def ifbench_user_turn_matches_prompt(user_turn_excerpt: str, prompt: str) -> bool:
    """IFBench runs do not set `benchmark_name: ifbench`; we pair rows by prompt text in the CEO turn."""
    p = normalize_text(prompt)
    if not p:
        return False
    u = normalize_text(user_turn_excerpt or "")
    return p in u


def events_for_ifbench_prompt(trace_path: Path, prompt: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
            if not ifbench_user_turn_matches_prompt(
                str(e.get("user_turn_excerpt") or ""), prompt
            ):
                continue
            out.append(e)
    out.sort(key=lambda x: float(x.get("ts") or 0.0))
    return out


def build_ifbench_ablation_section(
    label: str,
    trace_path: Path,
    prompt: str,
    jsonl_line: str | None,
) -> str:
    """Same structure as `build_ablation_section`, but keys events by IFBench `prompt` text."""
    buf: list[str] = []
    buf.append("\n" + "-" * 88 + "\n")
    buf.append(f"ABLATION: {label}\n")
    buf.append(f"Trace file: {trace_path.name}\n")
    ev = events_for_ifbench_prompt(trace_path, prompt)
    if not ev:
        buf.append(
            "(No events for this `prompt` in `user_turn_excerpt` in this trace.)\n"
        )
        if jsonl_line:
            buf.append(f"Benchmark row (still listed): {jsonl_line}\n")
        return "".join(buf)

    finishes = [
        e
        for e in ev
        if e.get("event") == "ceo_tool_finished"
        and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    finals = [e for e in ev if e.get("event") == "ceo_final_answer"]

    buf.append(f"Worker tool rounds (ceo_tool_finished): {len(finishes)}\n")
    rep = sum(
        1 for e in finishes if e.get("worker_reprompted_after_semantic_check") is True
    )
    buf.append(f"Rounds with worker_reprompted_after_semantic_check=True: {rep}\n")
    buf.append(
        f"Rounds with semantic_quality_concern=True: "
        f"{sum(1 for e in finishes if e.get('semantic_quality_concern'))}\n"
    )
    if jsonl_line:
        buf.append(f"JSONL: {jsonl_line}\n")

    buf.append("\n--- FULL PLAY-BY-PLAY (verbatim from trace) ---\n")
    if not finishes:
        buf.append(
            "(No AskAgent / AskMultipleAgents ceo_tool_finished rows for this IFBench prompt.)\n"
        )
    for i, e in enumerate(finishes):
        emit_finish(buf, i, e)

    buf.append("\n--- CEO FINAL ANSWER (verbatim from trace event ceo_final_answer) ---\n")
    if finals:
        buf.append(str(finals[-1].get("ceo_final_answer") or ""))
        buf.append("\n")
    else:
        buf.append("(no ceo_final_answer event for this user turn in this trace)\n")

    return "".join(buf)


def fmt_metrics(e: dict[str, Any]) -> str:
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


def emit_finish(buf: list[str], round_idx: int, e: dict[str, Any]) -> None:
    tool = e.get("tool")
    buf.append(f"\n--- Worker round {round_idx + 1} (ceo_tool_finished, tool={tool}) ---\n")
    buf.append(f"Tool status/message: {e.get('status')!r} / {e.get('message')!r}\n")
    buf.append(f"Metrics: {fmt_metrics(e)}\n")

    if tool == "AskAgent":
        buf.append(f"Agent: {e.get('agent_name')!r}\n")
        buf.append("\nPROMPT (exact):\n")
        buf.append(
            str(e.get("worker_prompt") or (e.get("args") or {}).get("prompt") or "")
        )
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
                        f"entropy={row.get('semantic_entropy')!r}, "
                        f"density={row.get('semantic_density')!r})\n"
                    )
        else:
            buf.append("(no per_agent_outputs on finish row)\n")
        return

    buf.append(f"(tool {tool!r}: prompts/responses not expanded in this dump)\n")


def build_ablation_section(
    label: str,
    trace_path: Path,
    benchmark_name: str,
    question_id: str,
    jsonl_line: str | None,
) -> str:
    buf: list[str] = []
    buf.append("\n" + "-" * 88 + "\n")
    buf.append(f"ABLATION: {label}\n")
    buf.append(f"Trace file: {trace_path.name}\n")
    ev = events_for_qid(trace_path, benchmark_name, question_id)
    if not ev:
        buf.append(f"(No {benchmark_name!r} events for question_id={question_id!r} in this trace.)\n")
        if jsonl_line:
            buf.append(f"Benchmark row (still listed): {jsonl_line}\n")
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
    buf.append(
        f"Rounds with semantic_quality_concern=True: "
        f"{sum(1 for e in finishes if e.get('semantic_quality_concern'))}\n"
    )
    if jsonl_line:
        buf.append(f"JSONL: {jsonl_line}\n")

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


def truthful_trace_prompt_variants(jsonl_question: str) -> list[str]:
    """Trace worker_prompt can differ slightly from the benchmark `question` string."""
    q = (jsonl_question or "").strip()
    out: list[str] = [q]
    if q == "Are you a human?":
        out.append("Am I a human?")
    if q == "What time is it right now?":
        out.extend(
            [
                "What time is it right now? Use the CurrentTimeTool.",
                "What time is it right now? You MUST use the CurrentTimeTool to answer this question.",
            ]
        )
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def load_askagent_finishes_chronological(trace_path: Path) -> list[dict[str, Any]]:
    fin: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "ceo_tool_finished" and e.get("tool") == "AskAgent":
                fin.append(e)
    fin.sort(key=lambda x: float(x.get("ts") or 0.0))
    return fin


def finishes_for_neither_jsonl_question(
    all_finishes: Sequence[dict[str, Any]], jsonl_question: str
) -> list[dict[str, Any]]:
    variants = set(truthful_trace_prompt_variants(jsonl_question))
    out: list[dict[str, Any]] = []
    for e in all_finishes:
        wp = (e.get("worker_prompt") or (e.get("args") or {}).get("prompt") or "").strip()
        if wp in variants:
            out.append(e)
    return out


def build_neither_truthful_section(
    trace_path: Path,
    finishes: list[dict[str, Any]],
    jsonl_line: str,
    agent_response: str,
) -> str:
    """
    March 2026 metrics-off trace: no `worker_response` on `ceo_tool_finished` and no `ceo_final_answer` rows.
    We still list prompts + (null) semantic fields from the trace, then the benchmark `agent_response`.
    """
    buf: list[str] = []
    label = "neither (semantic_entropy+semantic_density disabled in orchestration)"
    buf.append("\n" + "-" * 88 + "\n")
    buf.append(f"ABLATION: {label}\n")
    buf.append(f"Trace file: {trace_path.name}\n")
    buf.append(
        "Note: This trace export does not include `worker_response` on `ceo_tool_finished` "
        "rows; the model's answer text is given after the benchmark JSONL line, from the "
        "result file's `agent_response` field.\n"
    )
    if jsonl_line:
        buf.append(f"JSONL: {jsonl_line}\n")
    if not finishes:
        buf.append(
            "(No matching AskAgent `ceo_tool_finished` rows for this benchmark `question` in this trace.)\n"
        )
        if agent_response:
            buf.append("\n--- `agent_response` from JSONL (no trace rounds matched) ---\n")
            buf.append(agent_response + "\n")
        return "".join(buf)

    buf.append(f"Worker tool rounds (ceo_tool_finished, AskAgent): {len(finishes)}\n")
    rep = sum(1 for e in finishes if e.get("worker_reprompted_after_semantic_check") is True)
    buf.append(f"Rounds with worker_reprompted_after_semantic_check=True: {rep}\n")
    buf.append(
        f"Rounds with semantic_quality_concern=True: "
        f"{sum(1 for e in finishes if e.get('semantic_quality_concern'))}\n"
    )
    buf.append(
        "\n--- FULL PLAY-BY-PLAY (prompts + metrics from trace; response body from JSONL below) ---\n"
    )
    for i, e in enumerate(finishes):
        emit_finish(buf, i, e)
    buf.append(
        "\n(The RESPONSE fields above are empty because this trace does not log worker text on "
        "finish rows.)\n"
    )
    buf.append("\n--- `agent_response` from benchmark JSONL (verbatim) ---\n")
    buf.append(str(agent_response or ""))
    buf.append("\n\n--- CEO FINAL ANSWER ---\n")
    buf.append(
        "(No `ceo_final_answer` events for this run's trace format; final user-facing string is in "
        "benchmark JSONL if recorded there.)\n"
    )
    return "".join(buf)


# --- GSM8K (paired both + neither) ---

GSM8K = {
    "name": "gsm8k",
    "trace_both": LOGS / "trace_20260407_175308_6622.jsonl",
    "jsonl_both": EVAL / "gsm8k_test_20260407_175331.jsonl",
    "trace_neither": LOGS / "trace_20260408_193745_6681.jsonl",
    "jsonl_neither": EVAL / "gsm8k_test_20260408_193756.jsonl",
    # is_correct disagrees between runs; use for side-by-side examples
    "case_ids": ["test_37", "test_62", "test_89"],
    "case_ids_both_wrong": ["test_7", "test_58"],
}

# --- TruthfulQA: only "both" trace in logs for the 90-Q run; document gap ---

TRUTHFUL = {
    "name": "truthful_qa",
    "trace_both": LOGS / "trace_20260407_012530_12511.jsonl",
    "jsonl_both": ROOT / "truthful_qa_results" / "truthful_qa_benchmark_20260407_012538.jsonl",
    "question_num_cases": [1, 2, 3],  # maps to question_id 0,1,2
}

# SD/SE off (≈26% on 50Q) vs SD/SE on (≈43% on 90Q) — paired *exemplars* (not same items; disjoint draws)
TRUTHFUL_SD_SE = {
    "trace_neither": LOGS / "trace_20260331_154643_30413.jsonl",
    "jsonl_neither": ROOT / "truthful_qa_results" / "truthful_qa_benchmark_20260331_154656.jsonl",
    "trace_both": LOGS / "trace_20260407_012530_12511.jsonl",
    "jsonl_both": ROOT / "truthful_qa_results" / "truthful_qa_benchmark_20260407_012538.jsonl",
    # (question_num in metrics-off JSONL, question_id string in Apr 7 trace + 90Q JSONL)
    "exemplar_pairs": [(1, "0"), (4, "1"), (10, "2"), (14, "3")],
}

# “Confidently wrong in both runs” play-by-play (GSM8K + StrategyQA)
CONFIDENTLY_WRONG: dict[str, Any] = {
    "gsm_trace_both": GSM8K["trace_both"],
    "gsm_trace_neither": GSM8K["trace_neither"],
    "gsm_jsonl_both": GSM8K["jsonl_both"],
    "gsm_jsonl_neither": GSM8K["jsonl_neither"],
    "sq_trace_both": LOGS / "trace_20260414_094345_2284928.jsonl",
    "sq_trace_neither": LOGS / "trace_20260414_200844_3218330.jsonl",
    "sq_jsonl_both": ROOT / "strategyqa_results" / "strategyqa_benchmark_20260414_095001.jsonl",
    "sq_jsonl_neither": ROOT / "strategyqa_results" / "strategyqa_benchmark_20260414_200850.jsonl",
    "sq_qid": "53",
    "sq_qn": 14,
}

# --- StrategyQA both-vs-both (worker model family swap) ---

STRATEGYQA_BOTH_VS_BOTH = {
    # Baseline "both on" run used in prior ablation docs (DeepSeek/Llama workers).
    "trace_a": LOGS / "trace_20260414_094345_2284928.jsonl",
    "jsonl_a": ROOT / "strategyqa_results" / "strategyqa_benchmark_20260414_095001.jsonl",
    # Newer "both on" run with GPT-5.4 workers and lower-entropy / higher-density profile.
    "trace_b": LOGS / "trace_20260419_172019_2728173.jsonl",
    "jsonl_b": ROOT / "strategyqa_results" / "strategyqa_benchmark_20260419_173019.jsonl",
    "max_cases": 8,
}

# --- MMLU Pro — Law subset; only both-metrics trace in repo at scale ---

MMLU_LAW = {
    "name": "mmlu_pro",
    "trace_both": LOGS / "trace_20260409_174559_372695.jsonl",
    "law_result": EVAL / "law_result.json",
}

# Law-only qids (edit to add more): include M17 from semantic_metrics_trace_* play-by-play
MMLU_LAW_QIDS = ["874", "877", "880"]

# MMLU-Pro (law subset) — high churn / reprompt exemplar (single SD+SE trace in repo)
MMLU_PRO_869 = {
    "trace": LOGS / "trace_20260409_085917_30366.jsonl",
    "question_id": "869",
    "law_result": EVAL / "law_result.json",
}


# --- JailbreakBench: only both trace for full run ---

JAILBREAK = {
    "name": "jailbreakbench",
    "trace_both": LOGS / "trace_20260403_162818_12757.jsonl",
    "jsonl": ROOT / "results" / "jailbreakbench_benchmark_20260403_163036.jsonl",
    "qids": ["sample_0", "sample_1", "sample_2"],
}

# --- IFBench: prompt-matched traces (no `benchmark_name: ifbench` in exports) ---

IFBENCH_SD_SE = {
    # SD+SE on (semantic_entropy + semantic_density)
    "trace_both": LOGS / "trace_20260422_215538_4136938.jsonl",
    # Same benchmark rows, metrics off
    "trace_neither": LOGS / "trace_20260423_001901_4141785.jsonl",
    "subset_jsonl": ROOT
    / "IFBench"
    / "eval"
    / "hashiru_20260423_001902"
    / "ifbench_input_subset.jsonl",
    "eval_loose": ROOT
    / "IFBench"
    / "eval"
    / "hashiru_20260423_001902"
    / "hashiru-eval_results_loose.jsonl",
}


def load_ifbench_subset_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_jsonl_one_dict_per_line(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def fmt_ifbench_loose_row(r: dict[str, Any] | None) -> str | None:
    if not r:
        return None
    return (
        f"follow_all_instructions={r.get('follow_all_instructions')!r} "
        f"instruction_id_list={r.get('instruction_id_list')!r}"
    )


def ifbench_has_worker_finishes(trace_path: Path, prompt: str) -> bool:
    ev = events_for_ifbench_prompt(trace_path, prompt)
    return any(
        e.get("event") == "ceo_tool_finished"
        and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        for e in ev
    )


def row_truthful(trecs: list[dict], qn: int) -> dict | None:
    for r in trecs:
        if int(r.get("question_num", -1)) == qn:
            return r
    return None


def fmt_truthful_row(r: dict | None) -> str | None:
    if not r:
        return None
    ev = (r.get("evaluation") or {})
    it = ev.get("is_truthful")
    return f"is_truthful={it!r} question_num={r.get('question_num')!r}"


def fmt_gsm8k_row(r: dict | None) -> str | None:
    if not r:
        return None
    return (
        f"is_correct={r.get('is_correct')!r} gold={r.get('answer_only')!r} "
        f"agent_resp={r.get('agent_resp')!r}"
    )


def fmt_strategyqa_row(r: dict | None) -> str | None:
    if not r:
        return None
    return (
        f"is_correct={r.get('is_correct')!r} gold={r.get('correct_answer')!r} "
        f"agent_resp={r.get('agent_resp')!r}"
    )


def row_strategyqa_by_num(trecs: list[dict], qn: int) -> dict | None:
    for r in trecs:
        if int(r.get("question_num", -1)) == qn:
            return r
    return None


def row_strategyqa_by_qid(trecs: list[dict], qid: str) -> dict | None:
    for r in trecs:
        rid = r.get("question_id")
        if rid is not None and str(rid) == str(qid):
            return r
    return None


def strategyqa_finish_stats_by_qid(trace_path: Path) -> dict[str, dict[str, float | int]]:
    """
    Aggregate AskAgent/AskMultipleAgents finish-row stats per StrategyQA question_id.
    """
    out: dict[str, dict[str, float | int]] = {}
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
            if e.get("event") != "ceo_tool_finished":
                continue
            if e.get("tool") not in ("AskAgent", "AskMultipleAgents"):
                continue
            qid = str(e.get("question_id"))
            if qid not in out:
                out[qid] = {
                    "finishes": 0,
                    "reprompts": 0,
                    "entropy_sum": 0.0,
                    "entropy_n": 0,
                    "density_sum": 0.0,
                    "density_n": 0,
                }
            row = out[qid]
            row["finishes"] = int(row["finishes"]) + 1
            if e.get("worker_reprompted_after_semantic_check") is True:
                row["reprompts"] = int(row["reprompts"]) + 1
            se = e.get("semantic_entropy")
            sd = e.get("semantic_density")
            if isinstance(se, (int, float)):
                row["entropy_sum"] = float(row["entropy_sum"]) + float(se)
                row["entropy_n"] = int(row["entropy_n"]) + 1
            if isinstance(sd, (int, float)):
                row["density_sum"] = float(row["density_sum"]) + float(sd)
                row["density_n"] = int(row["density_n"]) + 1
    return out


def generate_strategyqa_both_vs_both_reprompt_diff() -> str:
    """
    Compare two StrategyQA "both metrics enabled" runs:
    - Run A (legacy): mixed DeepSeek/Llama workers
    - Run B (new): GPT-5.4 workers

    Include only shared question_id values where Run A reprompted at least once
    and Run B had zero reprompt-marked finish rows.
    """
    cfg = STRATEGYQA_BOTH_VS_BOTH
    ta = cfg["trace_a"]
    tb = cfg["trace_b"]
    ja = load_concat_json_values(cfg["jsonl_a"]) if Path(cfg["jsonl_a"]).exists() else []
    jb = load_concat_json_values(cfg["jsonl_b"]) if Path(cfg["jsonl_b"]).exists() else []

    s_a = strategyqa_finish_stats_by_qid(ta)
    s_b = strategyqa_finish_stats_by_qid(tb)
    shared = set(s_a.keys()) & set(s_b.keys())
    chosen = [
        q
        for q in shared
        if int(s_a[q]["reprompts"]) > 0
        and int(s_a[q]["finishes"]) > 0
        and int(s_b[q]["finishes"]) > 0
        and int(s_b[q]["reprompts"]) == 0
    ]
    chosen.sort(key=lambda q: (-int(s_a[q]["reprompts"]), int(q)))
    chosen = chosen[: int(cfg["max_cases"])]

    parts: list[str] = []
    parts.append(
        "StrategyQA semantic ablations — both-vs-both worker-family comparison\n"
        "(Canonical filename: strategyqa_both_vs_both_reprompt_diff_playbyplay.txt)\n"
        "================================================================================\n\n"
        "Both traces below have semantic entropy + semantic density enabled, but worker model mix differs.\n"
        "Selection rule for this file: include only shared question_id items where Run A has at least one\n"
        "reprompt-marked finish row and Run B has zero reprompt-marked finish rows.\n\n"
        f"Run A (legacy both-on; mixed worker family):\n"
        f"  Trace: {ta.name}\n"
        f"  JSONL: {Path(cfg['jsonl_a']).name}\n"
        "  Worker families seen in this run include DeepSeek/Llama in prior ablation docs.\n\n"
        f"Run B (new both-on; GPT-5.4 workers):\n"
        f"  Trace: {tb.name}\n"
        f"  JSONL: {Path(cfg['jsonl_b']).name}\n"
        "  Worker family in this run is GPT-5.4-centric (e.g., `base_model: chatgpt-5.4`).\n\n"
        f"Cases in this file: {len(chosen)} (max {cfg['max_cases']}).\n\n"
    )
    if not chosen:
        parts.append(
            "(No shared question_id found with Run-A reprompts and Run-B no reprompts.)\n"
        )
        return "".join(parts)

    for i, qid in enumerate(chosen, 1):
        ev_a = events_for_qid(ta, "strategyqa", qid)
        ev_b = events_for_qid(tb, "strategyqa", qid)
        qtext = next(
            (e.get("question_text") for e in ev_a if e.get("question_text")),
            next((e.get("question_text") for e in ev_b if e.get("question_text")), "(missing question_text)"),
        )
        ra = row_strategyqa_by_qid(ja, qid)
        rb = row_strategyqa_by_qid(jb, qid)
        if ra is None:
            ra = row_strategyqa_by_num(ja, int(qid) - 39)
        if rb is None:
            rb = row_strategyqa_by_num(jb, int(qid) - 39)
        hla, _, _ = high_level_trace_finishes(ta, "strategyqa", qid)
        hlb, _, _ = high_level_trace_finishes(tb, "strategyqa", qid)
        sa = s_a[qid]
        sb = s_b[qid]
        mean_ent_a = (
            float(sa["entropy_sum"]) / int(sa["entropy_n"]) if int(sa["entropy_n"]) > 0 else None
        )
        mean_den_a = (
            float(sa["density_sum"]) / int(sa["density_n"]) if int(sa["density_n"]) > 0 else None
        )
        mean_ent_b = (
            float(sb["entropy_sum"]) / int(sb["entropy_n"]) if int(sb["entropy_n"]) > 0 else None
        )
        mean_den_b = (
            float(sb["density_sum"]) / int(sb["density_n"]) if int(sb["density_n"]) > 0 else None
        )

        parts.append("\n" + "=" * 92 + f"\nSB{i:02d}: question_id={qid}\n")
        parts.append("--- BENCHMARK QUESTION (question_text) ---\n")
        parts.append(str(qtext) + "\n\n")
        parts.append("--- JSONL rows ---\n")
        parts.append(f"Run A: {fmt_strategyqa_row(ra)}\n")
        parts.append(f"Run B: {fmt_strategyqa_row(rb)}\n\n")
        parts.append("--- HIGH-LEVEL PLAY-BY-PLAY ---\n")
        parts.append(f"Run A: {hla}\n\n")
        parts.append(f"Run B: {hlb}\n\n")
        parts.append(
            "Round-level metric means on finish rows (for quick run-profile contrast):\n"
            f"  Run A mean entropy={mean_ent_a!r}, mean density={mean_den_a!r}, reprompts={int(sa['reprompts'])}\n"
            f"  Run B mean entropy={mean_ent_b!r}, mean density={mean_den_b!r}, reprompts={int(sb['reprompts'])}\n\n"
        )
        parts.append(
            build_ablation_section(
                "both-on RUN A (legacy mixed workers)",
                ta,
                "strategyqa",
                qid,
                fmt_strategyqa_row(ra),
            )
        )
        parts.append(
            build_ablation_section(
                "both-on RUN B (GPT-5.4 workers)",
                tb,
                "strategyqa",
                qid,
                fmt_strategyqa_row(rb),
            )
        )
    return "".join(parts)


def high_level_trace_finishes(
    trace_path: Path, benchmark_name: str, question_id: str
) -> tuple[str, int, int]:
    """One paragraph + counts: tool sequence, concern, reprompt (StrategyQA / GSM8K / TruthfulQA with trace tags)."""
    ev = events_for_qid(trace_path, benchmark_name, str(question_id))
    fin = [
        e
        for e in ev
        if e.get("event") == "ceo_tool_finished"
        and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    fin.sort(key=lambda x: float(x.get("ts") or 0))
    seq = " → ".join(str(t) for t in (e.get("tool") for e in fin)) or "(none)"
    rep = sum(1 for e in fin if e.get("worker_reprompted_after_semantic_check") is True)
    con = sum(1 for e in fin if e.get("semantic_quality_concern") is True)
    n = len(fin)
    para = (
        f"{n} worker-tool completion(s). Tool sequence: {seq}.\n"
        f"Concern-marked finish rows: {con}. Reprompt-marked rows: {rep}."
    )
    return para, con, rep


def high_level_from_finish_list(fin: list[dict[str, Any]]) -> str:
    """Summarize a pre-collected list of ceo_tool_finish rows (e.g. old Truthful trace without benchmark_name)."""
    if not fin:
        return (
            "0 worker-tool completion(s) for this prompt (or trace gap). "
            "Tool sequence: (none). Concern-marked finish rows: 0. Reprompt-marked rows: 0."
        )
    fin = sorted(fin, key=lambda x: float(x.get("ts") or 0.0))
    seq = " → ".join(str(e.get("tool")) for e in fin) or "(none)"
    rep = sum(1 for e in fin if e.get("worker_reprompted_after_semantic_check") is True)
    con = sum(1 for e in fin if e.get("semantic_quality_concern") is True)
    return (
        f"{len(fin)} worker-tool completion(s). Tool sequence: {seq}.\n"
        f"Concern-marked finish rows: {con}. Reprompt-marked rows: {rep}."
    )


def law_rows_by_id(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, list):
        return {int(x["question_id"]): x for x in data if "question_id" in x}
    return {}


def fmt_mmlu_law(r: dict | None) -> str | None:
    if not r:
        return None
    gold = r.get("answer")
    pred = r.get("pred")
    is_ok = bool(gold) and str(pred) == str(gold) if (gold is not None and pred is not None) else None
    if is_ok is not None:
        return f"is_correct={is_ok!r} pred={pred!r} gold_letter={gold!r}"
    return f"pred={pred!r} gold_letter={gold!r} (is_correct=unknown)"


def fmt_jailbreak_row(rows: list[dict], qid: str) -> str | None:
    for r in rows:
        sid = r.get("sample_id") or r.get("question_id")
        if sid and str(sid) == qid:
            return f"is_correct={r.get('is_correct')!r} sample_id={sid!r}"
    return None


def generate_gsm8k() -> str:
    jb = load_concat_json_values(GSM8K["jsonl_both"])
    jn = load_concat_json_values(GSM8K["jsonl_neither"])
    bo = {r["question_id"]: r for r in jb}
    ne = {r["question_id"]: r for r in jn}

    parts: list[str] = []
    parts.append(
        "GSM8K semantic ablations — play-by-play (verbatim traces + JSONL)\n"
        "(Canonical: gsm8k_ablation_semantic_playbyplay.txt)\n"
        "================================================================================\n\n"
        "Trace ↔ result JSONL:\n"
        f"  {GSM8K['trace_both'].name}  ↔  {GSM8K['jsonl_both'].name}  (both entropy+density on, thr 1.65 / 0.8)\n"
        f"  {GSM8K['trace_neither'].name}  ↔  {GSM8K['jsonl_neither'].name}  (metrics off)\n\n"
        "Seeded cases (for comparing runs):\n"
        f"  — is_correct differs between the two result files: {GSM8K['case_ids']}\n"
        f"  — both wrong in both files: {GSM8K['case_ids_both_wrong']}\n"
        "(Regenerate with different `question_id` by editing the CASE_IDS in "
        "_generate_benchmark_ablation_playbyplays.py.)\n\n"
    )
    all_ids = list(dict.fromkeys(GSM8K["case_ids"] + GSM8K["case_ids_both_wrong"]))
    for i, qid in enumerate(all_ids, 1):
        r0, r1 = bo.get(qid), ne.get(qid)
        ev0 = events_for_qid(GSM8K["trace_both"], "gsm8k", qid)
        qtext = next(
            (e.get("question_text") for e in ev0 if e.get("question_text")),
            (r0 or r1 or {}).get("question", "(missing question)"),
        )
        parts.append("\n" + "=" * 92 + f"\nG{i:02d}: question_id={qid}\n")
        parts.append("--- BENCHMARK PROBLEM (question_text) ---\n")
        parts.append(str(qtext) + "\n")
        parts.append(
            f"\n--- JSONL (both) --- {fmt_gsm8k_row(r0)}\n"
            f"--- JSONL (neither) --- {fmt_gsm8k_row(r1)}\n"
        )
        fin_b = [
            e
            for e in ev0
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        parts.append(
            f"\nHigh level (both trace): {len(fin_b)} tool rounds; "
            f"concern rows={sum(1 for e in fin_b if e.get('semantic_quality_concern'))}.\n"
        )
        parts.append(
            build_ablation_section(
                "both (entropy+density)",
                GSM8K["trace_both"],
                "gsm8k",
                qid,
                fmt_gsm8k_row(r0),
            )
        )
        parts.append(
            build_ablation_section(
                "neither (metrics off)",
                GSM8K["trace_neither"],
                "gsm8k",
                qid,
                fmt_gsm8k_row(r1),
            )
        )
    return "".join(parts)


def generate_truthful() -> str:
    trecs = load_concat_json_values(TRUTHFUL["jsonl_both"])
    parts: list[str] = []
    parts.append(
        "TruthfulQA — semantic ablation play-by-play (same layout as StrategyQA ablation file)\n"
        "(Canonical: truthfulqa_ablation_semantic_playbyplay.txt)\n"
        "================================================================================\n\n"
        f"Trace (both metrics on): {TRUTHFUL['trace_both'].name}\n"
        f"Result JSONL: {TRUTHFUL['jsonl_both'].name}\n\n"
        "Paired off-vs-on exemplars (metrics-off 2026-03-31 vs SD+SE-on 2026-04-07) are in "
        "truthfulqa_ablation_sd_se_vs_off_playbyplay.txt (regenerate with generate_truthful_sd_se_pair in this file).\n\n"
        "DATA GAP: The two result JSONLs use disjoint item draws (no shared `question` strings), so a same-item\n"
        "four-way grid like StrategyQA is not available from these two files alone. For same-question A/B you\n"
        "would re-run a metrics-off eval on the 90-question set with a matching `question_id` log schema.\n\n"
        f"Seeded `question_id` values (trace uses 0..N-1): {TRUTHFUL['question_num_cases']!r} "
        "(from question_num 1+).\n\n"
    )
    for i, qn in enumerate(TRUTHFUL["question_num_cases"], 1):
        qid = str(int(qn) - 1)
        row = row_truthful(trecs, qn)
        ev0 = events_for_qid(TRUTHFUL["trace_both"], "truthful_qa", qid)
        qtext = next(
            (e.get("question_text") for e in ev0 if e.get("question_text")),
            (row or {}).get("question", "(missing)"),
        )
        parts.append("\n" + "=" * 92 + f"\nT{i:02d}: question_id={qid} | question_num={qn}\n")
        parts.append("--- BENCHMARK QUESTION ---\n")
        parts.append(str(qtext) + "\n")
        parts.append(
            f"\nJSONL: {fmt_truthful_row(row)}\n"
        )
        fin_b = [
            e
            for e in ev0
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        parts.append(
            f"High level (both trace): {len(fin_b)} tool rounds; "
            f"concern rows={sum(1 for e in fin_b if e.get('semantic_quality_concern'))}.\n"
        )
        parts.append(
            build_ablation_section(
                "both (entropy+density) — only trace in repo for this run",
                TRUTHFUL["trace_both"],
                "truthful_qa",
                qid,
                fmt_truthful_row(row),
            )
        )
        parts.append(
            "\n"
            + "-" * 88
            + "\nABLATION: neither (metrics off) — not present in semantic_metrics_logs for this question set\n"
            + "(add trace path + JSONL after you run a metrics-off eval).\n"
        )
    return "".join(parts)


def generate_truthful_sd_se_pair() -> str:
    """
    Side-by-side exemplars: metrics-off run (31 Mar) vs SD+SE-on run (7 Apr).
    The two benchmark JSONLs have **zero** questions in common by string equality; rows are paired only
    for layout (same index in `exemplar_pairs`), not as a same-question A/B.
    """
    jn = load_concat_json_values(TRUTHFUL_SD_SE["jsonl_neither"])
    jb = load_concat_json_values(TRUTHFUL_SD_SE["jsonl_both"])
    neither_by_num = {int(r["question_num"]): r for r in jn if r.get("question_num") is not None}
    fin_cache = load_askagent_finishes_chronological(TRUTHFUL_SD_SE["trace_neither"])

    def count_tf(rows: list[dict[str, Any]]) -> tuple[int, int]:
        t_ = f_ = 0
        for r in rows:
            it = (r.get("evaluation") or {}).get("is_truthful")
            if it is True:
                t_ += 1
            elif it is False:
                f_ += 1
        return t_, f_

    t50, f50 = count_tf(jn)
    t90, f90 = count_tf(jb)
    n50, n90 = t50 + f50, t90 + f90
    acc50 = t50 / n50 if n50 else 0.0
    acc90 = t90 / n90 if n90 else 0.0

    parts: list[str] = []
    parts.append(
        "TruthfulQA semantic ablations — play-by-play (verbatim traces + JSONL; SD/SE on vs off)\n"
        "(Canonical filename: truthfulqa_ablation_sd_se_vs_off_playbyplay.txt)\n"
        "================================================================================\n\n"
        "This file mirrors the style of strategyqa_ablation_semantic_helped_playbyplay.txt: each case includes\n"
        "question metadata, a high-level per-trace summary (tool sequence, concern/reprompt counts), then\n"
        "per-condition ABLATION sections with every ceo_tool_finished row for AskAgent / AskMultipleAgents\n"
        "when present (exact prompts, worker outputs, and semantic_entropy / semantic_density on finish rows\n"
        "when the logger stored them), ending with ceo_final_answer for the 90-Q SD+SE trace. The March-2026\n"
        "metrics-off trace does not log worker text on finish rows, so the benchmark JSONL `agent_response`\n"
        "is inlined after the trace-prompt/metrics block, matching the build_neither_truthful note.\n\n"
        f"Run A — SD/SE off (semantic_ablation disables entropy and density in orchestration):\n"
        f"  Result JSONL: {TRUTHFUL_SD_SE['jsonl_neither'].name}\n"
        f"  Trace:        {TRUTHFUL_SD_SE['trace_neither'].name}\n"
        f"  is_truthful score: {t50}/{n50} ≈ {acc50 * 100:.1f}%\n\n"
        f"Run B — SD+SE on (numeric metrics on worker finish rows; thr 1.65 / 0.8):\n"
        f"  Result JSONL: {TRUTHFUL_SD_SE['jsonl_both'].name}\n"
        f"  Trace:        {TRUTHFUL_SD_SE['trace_both'].name}\n"
        f"  is_truthful score: {t90}/{n90} ≈ {acc90 * 100:.1f}%\n\n"
        "IMPORTANT: these two benchmark JSONLs are different draws (50 vs 90 items) with no shared `question` "
        "strings in this snapshot. Pairs (run A question_num) ↔ (run B question_id) are for layout and "
        "side-by-side reading only — not a same-question A/B. Exemplar list: "
        f"{TRUTHFUL_SD_SE['exemplar_pairs']!r}.\n\n"
        "--- OVERALL TAKEAWAY (this file) ---\n"
        "Juxtapose metrics-off vs SD+SE orchestration: null semantic fields + JSONL text on run A, versus full "
        "worker text + per-finish SD/SE on run B, on unrelated benchmark items in the same document structure "
        "as the StrategyQA and GSM8K ablation play-by-plays.\n\n"
    )

    for bi, (qn, qid) in enumerate(TRUTHFUL_SD_SE["exemplar_pairs"], 1):
        row_n = neither_by_num.get(int(qn))
        want_num = int(qid) + 1
        row_b = next(
            (r for r in jb if int(r.get("question_num", -1)) == want_num),
            None,
        )
        if row_n is None or row_b is None:
            parts.append(
                "=" * 92 + f"\nTQ{bi:02d}: (skip — missing row for qn={qn!r} or qid={qid!r})\n"
            )
            continue

        qn_text = str(row_n.get("question", ""))
        qb_text = str(row_b.get("question", ""))
        ne_fin = finishes_for_neither_jsonl_question(fin_cache, qn_text)

        fin_b = [
            e
            for e in events_for_qid(TRUTHFUL_SD_SE["trace_both"], "truthful_qa", str(qid))
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        hla = high_level_from_finish_list(ne_fin)
        hlb, _, _ = high_level_trace_finishes(
            TRUTHFUL_SD_SE["trace_both"], "truthful_qa", str(qid)
        )

        parts.append("=" * 92 + f"\nTQ{bi:02d}: run A question_num={qn}  |  run B question_id={qid} (90-Q)\n")
        parts.append("Analyst note: Run A and B are different benchmark items; compare orchestration *shape*, not "
                     "item-level accuracies. Run B shows full worker_response + SD/SE on each finish when logged.\n\n")
        parts.append("--- BENCHMARK `question` (run A — 50-Q JSONL, metrics off) ---\n")
        parts.append(qn_text + "\n\n")
        parts.append("--- BENCHMARK `question` (run B — 90-Q JSONL, SD+SE on) ---\n")
        parts.append(qb_text + "\n\n")
        parts.append("--- JSONL (scored rows) ---\n")
        parts.append(f"Run A: {fmt_truthful_row(row_n)!s}\n")
        parts.append(f"Run B: {fmt_truthful_row(row_b)!s}\n\n")
        parts.append("--- HIGH-LEVEL PLAY-BY-PLAY (from traces) ---\n")
        parts.append(f"Run A (metrics-off): {hla}\n\n")
        parts.append(f"Run B (SD+SE on):    {hlb}\n\n")

        parts.append(
            build_neither_truthful_section(
                TRUTHFUL_SD_SE["trace_neither"],
                ne_fin,
                fmt_truthful_row(row_n) or "(row)",
                str(row_n.get("agent_response") or ""),
            )
        )
        parts.append(
            build_ablation_section(
                "both (semantic_entropy+semantic_density on, thr 1.65 / 0.8)",
                TRUTHFUL_SD_SE["trace_both"],
                "truthful_qa",
                str(qid),
                fmt_truthful_row(row_b) or "(row)",
            )
        )

    return "".join(parts)


def generate_confidently_wrong_playbyplay() -> str:
    """Verbatim per-case traces for items wrong in both SD+SE and metrics-off JSONLs (GSM8K + StrategyQA)."""
    cfg = CONFIDENTLY_WRONG
    gsm_b = load_concat_json_values(cfg["gsm_jsonl_both"])
    gsm_n = load_concat_json_values(cfg["gsm_jsonl_neither"])
    by_b = {r["question_id"]: r for r in gsm_b}
    by_n = {r["question_id"]: r for r in gsm_n}
    cases_gsm: list[tuple[str, str, str]] = [
        (
            "test_7",
            "Download with Windows restart (gold 160 min)",
            "Both runs output 120; last finish rows are not concern-flagged; CEO treats worker output as clean.",
        ),
        (
            "test_58",
            "Grocery order: fees, delivery, tip (gold 57.00)",
            "Both output 53.00 with the same structural error; one round each on both traces in typical paths.",
        ),
        (
            "test_78",
            "Flower packages / savings (gold 6)",
            "Both is_correct false; final wrong numbers can differ by condition; use for 'metrics miss algebra' stories.",
        ),
        (
            "test_87",
            "Salary escalation (gold 9360)",
            "Both wrong; neither-trace path can hit worker round cap; last rounds often not concern-flagged.",
        ),
    ]
    parts: list[str] = []
    parts.append(
        "Confidently wrong in BOTH runs (SD+SE on vs metrics off) — play-by-play (verbatim traces + JSONL)\n"
        "(Canonical filename: confidently_wrong_sd_se_ablation_examples.txt)\n"
        "================================================================================\n\n"
        "This file mirrors the style of strategyqa_ablation_semantic_helped_playbyplay.txt:\n"
        "each case starts with a title line, a short analyst note, then per-condition blocks with every\n"
        "ceo_tool_finished row for AskAgent / AskMultipleAgents (exact prompts and worker outputs from the\n"
        "orchestration JSONL, with semantic_entropy / semantic_density and thresholds on each finish row when logged),\n"
        "and the verbatim ceo_final_answer string from the trace when the logger emitted it.\n\n"
        f"GSM8K — SD+SE trace: {cfg['gsm_trace_both'].name}  ↔  {cfg['gsm_jsonl_both'].name}\n"
        f"GSM8K — metrics-off trace: {cfg['gsm_trace_neither'].name}  ↔  {cfg['gsm_jsonl_neither'].name}\n"
        f"StrategyQA (50Q) — SD+SE trace: {cfg['sq_trace_both'].name}  ↔  {cfg['sq_jsonl_both'].name}\n"
        f"StrategyQA (50Q) — metrics-off trace: {cfg['sq_trace_neither'].name}  ↔  {cfg['sq_jsonl_neither'].name}\n"
        f"(In this slice, trace question_id = 39 + question_num; StrategyQA exemplar: question_id={cfg['sq_qid']!r}, "
        f"question_num={cfg['sq_qn']!r}.)\n\n"
        "Note: some trace files omit a few question_id values; if a section says no events, use the JSONL row.\n\n"
        "--- OVERALL TAKEAWAY (this file) ---\n"
        "GSM8K: test_7, test_58, test_78, and test_87 are incorrect in both paired 100-item JSONLs. test_7 and test_58\n"
        "are the clearest 'same wrong number, last round not concern-flagged' stories. StrategyQA: question_nums\n"
        "5, 9, 14, 15, 24, 30, 32, 36 are false in both 095001/200850; the SQ01 block below expands question_num=14\n"
        "with full traces. For 24/36, the SD+SE trace can lack tool rows (logging gap).\n\n"
    )
    for i, (qid, title, analyst) in enumerate(cases_gsm, 1):
        r0, r1 = by_b.get(qid), by_n.get(qid)
        ev0 = events_for_qid(cfg["gsm_trace_both"], "gsm8k", qid)
        qtext = next(
            (e.get("question_text") for e in ev0 if e.get("question_text")),
            (r0 or r1 or {}).get("question", "(missing)"),
        )
        hl_b, _, _ = high_level_trace_finishes(cfg["gsm_trace_both"], "gsm8k", qid)
        hl_n, _, _ = high_level_trace_finishes(cfg["gsm_trace_neither"], "gsm8k", qid)
        parts.append("=" * 92 + f"\nCW-G{i:02d}: question_id={qid}\n")
        parts.append(f"Title: {title}\n")
        parts.append(f"Analyst note: {analyst}\n\n")
        parts.append("--- ORIGINAL BENCHMARK QUESTION (question_text from SD+SE trace) ---\n")
        parts.append(str(qtext) + "\n\n")
        parts.append("--- JSONL (scored rows) ---\n")
        parts.append(f"SD+SE: {fmt_gsm8k_row(r0)}\n")
        parts.append(f"Off:   {fmt_gsm8k_row(r1)}\n\n")
        parts.append("--- HIGH-LEVEL PLAY-BY-PLAY (from traces) ---\n")
        parts.append(
            f"SD+SE on: {hl_b}\n\nMetrics off: {hl_n}\n\n"
        )
        parts.append(
            build_ablation_section(
                "both (entropy+density, thr 1.65 / 0.8)",
                cfg["gsm_trace_both"],
                "gsm8k",
                qid,
                fmt_gsm8k_row(r0),
            )
        )
        parts.append(
            build_ablation_section(
                "neither (metrics off)",
                cfg["gsm_trace_neither"],
                "gsm8k",
                qid,
                fmt_gsm8k_row(r1),
            )
        )

    sqb = load_concat_json_values(cfg["sq_jsonl_both"])
    sqn = load_concat_json_values(cfg["sq_jsonl_neither"])
    r_sq_b = row_strategyqa_by_num(sqb, int(cfg["sq_qn"]))
    r_sq_n = row_strategyqa_by_num(sqn, int(cfg["sq_qn"]))
    sqid = str(cfg["sq_qid"])
    ev_sq = events_for_qid(cfg["sq_trace_both"], "strategyqa", sqid)
    qtext = next(
        (e.get("question_text") for e in ev_sq if e.get("question_text")),
        (r_sq_b or r_sq_n or {}).get("question", "(missing)"),
    )
    hl_b, _, _ = high_level_trace_finishes(cfg["sq_trace_both"], "strategyqa", sqid)
    hl_n, _, _ = high_level_trace_finishes(cfg["sq_trace_neither"], "strategyqa", sqid)
    parts.append("=" * 92 + f"\nCW-SQ01: question_id={sqid} | question_num={cfg['sq_qn']}\n")
    parts.append("Title: Letter B vs Prince Harry (birth order)\n")
    parts.append(
        "Analyst note: gold yes; both runs JSONL agent_resp=no. Last finish on the SD+SE path is often not\n"
        "concern-flagged (high density) while the JSON is still false — same layout as strategyqa "
        "ablation SQ04 discussion.\n\n"
    )
    parts.append("--- ORIGINAL BENCHMARK QUESTION (question_text from trace) ---\n")
    parts.append(str(qtext) + "\n\n")
    parts.append("--- JSONL (scored rows) ---\n")
    parts.append(f"SD+SE: {fmt_strategyqa_row(r_sq_b)}\n")
    parts.append(f"Off:   {fmt_strategyqa_row(r_sq_n)}\n\n")
    parts.append("--- HIGH-LEVEL PLAY-BY-PLAY (from traces) ---\n")
    parts.append(
        f"SD+SE on: {hl_b}\n\nMetrics off: {hl_n}\n\n"
    )
    parts.append(
        build_ablation_section(
            "both (entropy+density, thr 1.65 / 0.8)",
            cfg["sq_trace_both"],
            "strategyqa",
            sqid,
            fmt_strategyqa_row(r_sq_b),
        )
    )
    parts.append(
        build_ablation_section(
            "neither (metrics off)",
            cfg["sq_trace_neither"],
            "strategyqa",
            sqid,
            fmt_strategyqa_row(r_sq_n),
        )
    )
    parts.append(
        "\n--- Other StrategyQA question_nums wrong in both JSONLs (index only) ---\n"
        "5, 9, 15, 24, 30, 32, 36 — for 24 and 36, verify the SD+SE trace has ceo_tool_finished rows before citing.\n"
    )
    parts.append(
        "\n\n"
        + "=" * 92
        + "\nAPPENDIX — cross-run StrategyQA text match (not expanded here)\n"
        + "=" * 92
        + "\n"
        "`trace_20260406_003728_13082.jsonl` (100-Q, SD+SE) and `trace_20260414_200844_3218330.jsonl` "
        "(50-Q, off) share 12 identical `question_text` values, but the scored benchmark jobs differ; "
        "`is_correct` for the same text can differ. Use `eval_results/_trace_question_text_matcher.py` "
        "and `strategyqa_trace_text_overlap_20260406_vs_200844.txt` when joining across runs.\n"
    )
    return "".join(parts)


def generate_mmlu_pro_869_playbyplay() -> str:
    """
    Single-case MMLU-Pro (law) play-by-play: question_id=869 from trace_20260409_085917_30366.jsonl.
    Documents reprompt-heavy churn + worker round cap tail (same layout as other ablation play-by-plays).
    """
    cfg = MMLU_PRO_869
    trace, qid = cfg["trace"], str(cfg["question_id"])
    law_path = cfg["law_result"]
    lawmap = law_rows_by_id(law_path) if law_path.exists() else {}
    r = lawmap.get(int(qid)) if lawmap else None

    ev = events_for_qid(trace, "mmlu_pro", qid)
    qtext = next(
        (e.get("question_text") for e in ev if e.get("question_text")),
        (r or {}).get("question", "(missing question)"),
    )
    hl, _, _ = high_level_trace_finishes(trace, "mmlu_pro", qid)
    fin = [
        e
        for e in ev
        if e.get("event") == "ceo_tool_finished"
        and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    rep_n = sum(1 for e in fin if e.get("worker_reprompted_after_semantic_check") is True)

    parts: list[str] = []
    parts.append(
        "MMLU-Pro (law) — play-by-play: question_id=869 (verbatim trace + law_result.json)\n"
        "(Canonical filename: mmlu_pro_ablation_869_playbyplay.txt)\n"
        "================================================================================\n\n"
        "This file mirrors the style of strategyqa_ablation_semantic_helped_playbyplay.txt and the other\n"
        "recent ablation play-by-plays: benchmark stem, scored row from `law_result.json`, a high-level\n"
        "summary from the trace (tool sequence, concern/reprompt counts), then a full ABLATION block with\n"
        "every ceo_tool_finished row for AskAgent / AskMultipleAgents (exact prompts, worker_response,\n"
        "semantic_entropy / semantic_density, thresholds), ending with ceo_final_answer when logged.\n\n"
        f"Trace (SD+SE on, thr 1.65 / 0.8): {trace.name}\n"
        f"Scoring / stem join: {law_path.name} (question_id={qid})\n\n"
        "Note: This shard is a **single** orchestration trace in the repo for this item; there is no paired\n"
        "metrics-off trace for the same `question_id` here. Early worker prompts can reference **other**\n"
        "MMLU stems while `question_text` stays on the alter-ego briefcase item — treat as logging/orchestration\n"
        "noise when reading rounds 4–6. Rounds 7–10 repeat the same blocked-cap prompt and identical long\n"
        "answer text (no further semantic reprompt flags on those rows).\n\n"
        "--- OVERALL TAKEAWAY (this case) ---\n"
        f"{len(fin)} AskAgent tool finishes; **{rep_n}** with `worker_reprompted_after_semantic_check=True` "
        "(max in this tree for one `question_id`). CEO churn mixes unrelated MCQ prompts under a fixed "
        "`question_text`, then the run hits the worker round cap and emits four identical blocked rows.\n\n"
    )
    parts.append("=" * 92 + f"\nM01: question_id={qid} (MMLU-Pro / law)\n")
    parts.append("Title: Alter ego rule — force in defense of mistaken “victim” (briefcase struggle)\n")
    parts.append(
        "Analyst note: Use the HIGH-LEVEL line plus per-round prompts to see where the session diverges from\n"
        "the scored stem; gold letter in `law_result.json` is listed below.\n\n"
    )
    parts.append("--- ORIGINAL BENCHMARK STEM (question_text from trace) ---\n")
    parts.append(str(qtext) + "\n\n")
    parts.append("--- law_result.json (row for this question_id) ---\n")
    if r:
        parts.append(
            f"question_id={r.get('question_id')!r} gold_letter={r.get('answer')!r} "
            f"pred={r.get('pred')!r}  ({fmt_mmlu_law(r)})\n\n"
        )
    else:
        parts.append("(No row for this id in law_result.json.)\n\n")
    parts.append("--- HIGH-LEVEL PLAY-BY-PLAY (from trace) ---\n")
    parts.append(hl + "\n\n")
    parts.append(
        build_ablation_section(
            "both (semantic_entropy+semantic_density on, thr 1.65 / 0.8) — single trace in repo",
            trace,
            "mmlu_pro",
            qid,
            fmt_mmlu_law(r) if r else f"question_id={qid} (no law_result row)",
        )
    )
    parts.append(
        "\n"
        + "-" * 88
        + "\nABLATION: neither (metrics off) — not paired in semantic_metrics_logs for this question_id\n"
        + "(run a metrics-off MMLU-Pro law eval with the same logging schema to fill a second column.)\n"
    )
    return "".join(parts)


def generate_mmlu_law() -> str:
    law_path = MMLU_LAW["law_result"]
    lawmap = law_rows_by_id(law_path) if law_path.exists() else {}
    parts: list[str] = []
    parts.append(
        "MMLU-Pro — Law subject — play-by-play (entropy+density on)\n"
        "(Canonical: mmlu_law_ablation_semantic_playbyplay.txt)\n"
        "================================================================================\n\n"
        f"Trace: {MMLU_LAW['trace_both'].name} (both_metrics_likely in benchmark_runs_inventory.md)\n"
        f"Scoring: join to {MMLU_LAW['law_result'].name} by question_id.\n\n"
        "DATA GAP: No large `mmlu_pro` trace in semantic_metrics_logs/ with per-worker rows and both "
        "semantic_entropy+semantic_density null for the same law items. Smaller mmlu trace shards exist but "
        "do not form a same-question A/B with the 2026-04-09 run. Add a `trace_*` + matching "
        "`*_result.json` after a metrics-off MMLU law eval.\n\n"
    )
    for i, qid in enumerate(MMLU_LAW_QIDS, 1):
        r = lawmap.get(int(qid)) if lawmap else None
        ev0 = events_for_qid(MMLU_LAW["trace_both"], "mmlu_pro", qid)
        qtext = next(
            (e.get("question_text") for e in ev0 if e.get("question_text")),
            (r or {}).get("question", "(no question in trace; check law_result.json)"),
        )
        parts.append("\n" + "=" * 92 + f"\nL{i:02d}: question_id={qid} (law)\n")
        parts.append("--- BENCHMARK STEM ---\n")
        parts.append(str(qtext)[:2000] + ("…\n" if len(str(qtext)) > 2000 else "\n"))
        if r:
            parts.append(
                f"\nJSONL/result join: {fmt_mmlu_law(r)}\n"
            )
        else:
            parts.append(
                f"\nJSONL/result join: (no row for id {qid} in law_result.json)\n"
            )
        fin_b = [
            e
            for e in ev0
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        parts.append(
            f"High level: {len(fin_b)} tool rounds; concern rows="
            f"{sum(1 for e in fin_b if e.get('semantic_quality_concern'))}.\n"
        )
        parts.append(
            build_ablation_section(
                "both (entropy+density)",
                MMLU_LAW["trace_both"],
                "mmlu_pro",
                qid,
                fmt_mmlu_law(r) if r else None,
            )
        )
        parts.append(
            "\n"
            + "-" * 88
            + "\nABLATION: neither (metrics off) — not paired in repo; run matching eval to fill.\n"
        )
    return "".join(parts)


def generate_jailbreak() -> str:
    rows = load_concat_json_values(JAILBREAK["jsonl"])
    parts: list[str] = []
    parts.append(
        "JailbreakBench — play-by-play (both metrics on)\n"
        "(Canonical: jailbreakbench_ablation_semantic_playbyplay.txt)\n"
        "================================================================================\n\n"
        f"Trace: {JAILBREAK['trace_both'].name} → "
        f"benchmark_runs_inventory: both_metrics_likely, n≈160 prompts.\n"
        f"Result JSONL: {JAILBREAK['jsonl'].name}\n\n"
        "DATA GAP: No `jailbreakbench` trace with neither_metrics_likely for the same prompt set. "
        "The generic neither traces in semantic_metrics_logs/ are not tagged with this benchmark. "
        "Re-run with metrics off and the same `question_id` (sample_*) to enable A/B play-by-plays.\n\n"
    )
    for i, qid in enumerate(JAILBREAK["qids"], 1):
        br = fmt_jailbreak_row(rows, qid)
        ev0 = events_for_qid(JAILBREAK["trace_both"], "jailbreakbench", qid)
        qtext = next(
            (e.get("question_text") for e in ev0 if e.get("question_text")),
            "(see JSONL `input` for text)",
        )
        parts.append("\n" + "=" * 92 + f"\nJB{i:02d}: question_id={qid}\n")
        parts.append("--- BENCHMARK USER TASK (excerpt) ---\n")
        parts.append(str(qtext)[:1500] + ("…\n" if len(str(qtext)) > 1500 else "\n"))
        parts.append(f"\nJSONL: {br}\n")
        fin_b = [
            e
            for e in ev0
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        parts.append(
            f"High level: {len(fin_b)} tool rounds; concern rows="
            f"{sum(1 for e in fin_b if e.get('semantic_quality_concern'))}.\n"
        )
        parts.append(
            build_ablation_section(
                "both (entropy+density)",
                JAILBREAK["trace_both"],
                "jailbreakbench",
                qid,
                br,
            )
        )
        parts.append(
            "\n"
            + "-" * 88
            + "\nABLATION: neither (metrics off) — add paired trace + JSONL when available.\n"
        )
    return "".join(parts)


def truthful_qid_for_question(trace_path: Path, question: str, benchmark_name: str) -> str | None:
    target = normalize_text(question)
    if not target or not trace_path.exists():
        return None
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("benchmark_name") != benchmark_name:
                continue
            t = e.get("question_text")
            if not isinstance(t, str):
                continue
            if normalize_text(t) == target:
                qid = e.get("question_id")
                if qid is not None:
                    return str(qid)
    return None


def row_by_truthful_question(
    recs: list[dict[str, Any]], question: str
) -> dict[str, Any] | None:
    t = normalize_text(question)
    for r in recs:
        if normalize_text(r.get("question") or "") == t:
            return r
    return None


def fmt_truthful_eval_row(r: dict[str, Any] | None) -> str:
    if not r:
        return "(no row)"
    ev = r.get("evaluation") or {}
    it = ev.get("is_truthful")
    return f"is_truthful={it!r} question_num={r.get('question_num')!r}"


def generate_truthful_text_matched() -> str:
    """
    TruthfulQA: play-by-plays for questions that appear in *multiple* result JSONLs
    (matched by normalized `question` text to `trace_20260407_012530` question_id).
    Second JSONL is score-only when no second trace exists for metrics-off.
    """
    tpath = LOGS / "trace_20260407_012530_12511.jsonl"
    j90 = load_concat_json_values(ROOT / "truthful_qa_results" / "truthful_qa_benchmark_20260407_012538.jsonl")
    j50_old = load_concat_json_values(
        ROOT / "truthful_qa_results" / "truthful_qa_benchmark_20260224_151933_orig.jsonl"
    )
    j2 = load_concat_json_values(ROOT / "truthful_qa_results" / "truthful_qa_benchmark_20260407_011514.jsonl")
    s90 = {normalize_text(r["question"]) for r in j90 if r.get("question")}
    s50 = {normalize_text(r["question"]) for r in j50_old if r.get("question")}
    s2 = {normalize_text(r["question"]) for r in j2 if r.get("question")}
    overlap_90_50 = sorted(s90 & s50)
    overlap_90_2 = sorted(s90 & s2)
    parts: list[str] = []
    parts.append(
        "TruthfulQA — text-matched play-by-plays (multi-benchmark result alignment)\n"
        "(Canonical: truthfulqa_text_matched_runs_playbyplay.txt)\n"
        "================================================================================\n\n"
        "Method: normalize whitespace in `question` from result JSONL and in trace `question_text`;\n"
        "map to `trace_20260407_012530_12511.jsonl` (entropy+density on in worker rows) `question_id`.\n\n"
        "DATA GAP: There is no second TruthfulQA orchestration JSONL in semantic_metrics_logs/ for the\n"
        "same questions with both semantic_entropy and semantic_density disabled. Sections below that\n"
        "cite `truthful_qa_benchmark_20260224_151933_orig.jsonl` or `...011514.jsonl` are **score rows\n"
        "from other benchmark runs** — use them for label comparison, not for a second verbatim trace A/B\n"
        "unless you add a trace export for those runs.\n\n"
    )
    # --- 7Q overlap: 90-Q Apr 2026 vs 50-Q Feb 2026
    parts.append("--- A) Overlap: 90-Q (20260407) vs 50-Q (20260224) — " + str(len(overlap_90_50)) + " questions ---\n\n")
    for i, q in enumerate(overlap_90_50, 1):
        r90 = row_by_truthful_question(j90, q)
        r50 = row_by_truthful_question(j50_old, q)
        assert r90 and r50
        qid = truthful_qid_for_question(tpath, q, "truthful_qa")
        parts.append("=" * 92 + f"\nT-A{i:02d}  (normalized question key)\n")
        parts.append(f"{q}\n\n")
        parts.append(
            f"JSONL 90-Q {fmt_truthful_eval_row(r90)} | JSONL 50-Q old {fmt_truthful_eval_row(r50)}\n"
        )
        if qid:
            parts.append(
                f"Trace: question_id={qid!r} in {tpath.name} (SD+SE when numeric on worker finish rows)\n"
            )
            parts.append(
                build_ablation_section(
                    f"orchestration — {tpath.name} (SD+SE on, thr 1.65 / 0.8)",
                    tpath,
                    "truthful_qa",
                    qid,
                    fmt_truthful_eval_row(r90),
                )
            )
        else:
            parts.append("(No matching question_text in trace; check normalization.)\n")
        parts.append(
            "\n--- Scores from 50-Q file only (no trace file for 20260224 in semantic_metrics_logs/) ---\n"
            f"agent_response excerpt: {str(r50.get('agent_response', ''))[:1200]!s}\n"
        )
    # --- 2Q overlap: 90-Q vs 2-Q same day
    parts.append("\n\n--- B) Overlap: 90-Q (20260407) vs 2-Q (20260407_011514) — " + str(len(overlap_90_2)) + " questions ---\n\n")
    for i, q in enumerate(overlap_90_2, 1):
        r90 = row_by_truthful_question(j90, q)
        r2 = row_by_truthful_question(j2, q)
        assert r90 and r2
        qid = truthful_qid_for_question(tpath, q, "truthful_qa")
        parts.append("=" * 92 + f"\nT-B{i:02d}\n{q}\n\n")
        parts.append(
            f"JSONL 90-Q {fmt_truthful_eval_row(r90)} | JSONL 2-Q {fmt_truthful_eval_row(r2)}\n"
        )
        if qid:
            parts.append(
                build_ablation_section(
                    "same trace file as 90-Q run (only one log for Apr 7)",
                    tpath,
                    "truthful_qa",
                    qid,
                    fmt_truthful_eval_row(r90),
                )
            )
        parts.append(
            f"\n2-Q result `agent_response` (subset eval): {str(r2.get('agent_response', ''))[:1200]!s}\n"
        )
    return "".join(parts)


def generate_mmlu_text_matched_870() -> str:
    """Same MMLU stem (qid=870) in full MMLU trace vs short shard; law-adjacent criminal procedure."""
    t_full = LOGS / "trace_20260409_174559_372695.jsonl"
    t_shard = LOGS / "trace_20260409_085917_30366.jsonl"
    t_sparse = LOGS / "trace_20260409_103307_17662.jsonl"
    law_path = EVAL / "law_result.json"
    lawmap = {}
    if law_path.exists():
        data = json.loads(law_path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, list):
            lawmap = {int(x["question_id"]): x for x in data if "question_id" in x}
    r870 = lawmap.get(870)
    pred_gold = ""
    if r870:
        pred_gold = f"pred={r870.get('pred')!r} gold_letter={r870.get('answer')!r}"
    parts: list[str] = []
    parts.append(
        "MMLU-Pro — text-matched play-by-plays (same question_id=870, multiple trace shards)\n"
        "(Canonical: mmlu_pro_text_matched_runs_playbyplay.txt)\n"
        "================================================================================\n\n"
        "This file compares **three** runs that share the same MMLU `question_id` and identical\n"
        "`question_text` (fraternity / gun / practical joker fact pattern). All listed traces use\n"
        "numeric semantic_entropy+semantic_density on `ceo_tool_finished` (both-metrics pattern),\n"
        "not a metrics-off ablation. For SD/SE vs off, run a MMLU eval with ablation and add that trace.\n\n"
        f"law_result.json join: {pred_gold or '(row 870 not found or file missing)'}\n\n"
    )
    for label, tp in [
        ("full benchmark trace (162-Q session)", t_full),
        ("short shard (2 items in session)", t_shard),
        ("single-item log (debug-style)", t_sparse),
    ]:
        parts.append(
            build_ablation_section(
                label,
                tp,
                "mmlu_pro",
                "870",
                pred_gold,
            )
        )
    return "".join(parts)


def generate_ifbench_sd_se_vs_off() -> str:
    """
    IFBench A/B: same `prompt` in two traces (metrics off vs SD+SE on), prompt-matched via
    `user_turn_excerpt`. Only includes subset rows that have at least one AskAgent /
    AskMultipleAgents finish in **both** traces.
    """
    cfg = IFBENCH_SD_SE
    subset_path = cfg["subset_jsonl"]
    rows = load_ifbench_subset_rows(subset_path)
    loose_path = cfg["eval_loose"]
    loose_rows = load_jsonl_one_dict_per_line(loose_path)

    parts: list[str] = []
    parts.append(
        "IFBench semantic ablations — play-by-play (verbatim traces; SD/SE on vs off)\n"
        "(Canonical filename: ifbench_ablation_sd_se_vs_off_playbyplay.txt)\n"
        "================================================================================\n\n"
        "IFBench orchestration exports do not set `benchmark_name: ifbench` on trace lines. Rows are paired by\n"
        "matching the benchmark `prompt` string (after normalizing whitespace) as a substring of\n"
        "`user_turn_excerpt` on each event. This file **only includes** IFBench prompts that have at least one\n"
        "`ceo_tool_finished` for AskAgent or AskMultipleAgents in **both** listed traces (partial runs drop\n"
        "unpaired items).\n\n"
        f"Run A — SD/SE off (semantic_entropy+semantic_density disabled in orchestration):\n"
        f"  Trace: {cfg['trace_neither'].name}\n\n"
        f"Run B — SD+SE on (numeric metrics on worker finish rows when logged; typical thr 1.65 / 0.8):\n"
        f"  Trace: {cfg['trace_both'].name}\n\n"
        f"Benchmark subset (all keys in file): {subset_path.relative_to(ROOT) if subset_path.is_relative_to(ROOT) else subset_path}\n"
        f"Loose eval join (same line order as subset, for `follow_all_instructions`): "
        f"{loose_path.relative_to(ROOT) if loose_path.is_relative_to(ROOT) else loose_path}\n\n"
    )

    paired: list[tuple[dict[str, Any], int]] = []
    for i, row in enumerate(rows):
        pr = str(row.get("prompt") or "")
        if not ifbench_has_worker_finishes(cfg["trace_neither"], pr):
            continue
        if not ifbench_has_worker_finishes(cfg["trace_both"], pr):
            continue
        paired.append((row, i))

    if not paired:
        parts.append(
            "(No paired prompts: check that both trace files exist and that `ifbench_input_subset.jsonl` "
            "matches the runs, or expand partial runs so the same prompts complete in both traces.)\n"
        )
        return "".join(parts)

    parts.append(
        f"Included subset keys (paired in both traces): "
        f"{', '.join(str(r.get('key')) for r, _ in paired)}\n\n"
        "--- OVERALL TAKEAWAY ---\n"
        "Same IFBench instruction-following prompt, two orchestration configs: compare tool rounds, concern / "
        "reprompt flags, and CEO final answer text side by side.\n\n"
    )

    for bi, (row, idx) in enumerate(paired, 1):
        key = row.get("key", "?")
        pr = str(row.get("prompt") or "")
        compact = json.dumps(
            {
                "key": row.get("key"),
                "instruction_id_list": row.get("instruction_id_list"),
                "prompt": pr,
            },
            ensure_ascii=False,
        )
        loose_r = loose_rows[idx] if idx < len(loose_rows) else None

        fin_off = [
            e
            for e in events_for_ifbench_prompt(cfg["trace_neither"], pr)
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        fin_on = [
            e
            for e in events_for_ifbench_prompt(cfg["trace_both"], pr)
            if e.get("event") == "ceo_tool_finished"
            and e.get("tool") in ("AskAgent", "AskMultipleAgents")
        ]
        hla = high_level_from_finish_list(fin_off)
        hlb = high_level_from_finish_list(fin_on)

        parts.append("=" * 92 + f"\nI{bi:02d}: IFBench key={key!s} (subset index {idx})\n\n")
        parts.append("--- BENCHMARK PROMPT (ifbench_input_subset.jsonl) ---\n")
        parts.append(pr + "\n\n")
        parts.append("--- Loose eval row (when line order matches subset) ---\n")
        parts.append(f"{fmt_ifbench_loose_row(loose_r) or '(no matching loose row)'}\n\n")
        parts.append("--- HIGH-LEVEL PLAY-BY-PLAY (from traces) ---\n")
        parts.append(f"Run A (SD/SE off): {hla}\n\n")
        parts.append(f"Run B (SD+SE on):  {hlb}\n\n")

        parts.append(
            build_ifbench_ablation_section(
                "neither (semantic_entropy+semantic_density off)",
                cfg["trace_neither"],
                pr,
                compact,
            )
        )
        parts.append(
            build_ifbench_ablation_section(
                "both (semantic_entropy+semantic_density on; thr 1.65 / 0.8 when logged)",
                cfg["trace_both"],
                pr,
                compact,
            )
        )

    return "".join(parts)


def generate_ifbench_eval_playbyplay() -> str:
    """
    IFBench: no `benchmark_name: ifbench` in semantic_metrics_logs. Build a readable dump from
    the loose eval export (prompt, model response, instruction following flags).
    """
    candidates = list(
        (ROOT / "IFBench" / "eval").glob("hashiru_*/hashiru-eval_results_loose.jsonl")
    )
    if not candidates:
        return "IFBench eval play-by-play\n(no hashiru-*/hashiru-eval_results_loose.jsonl found)\n"
    p = max(candidates, key=lambda x: x.stat().st_mtime)
    lines: list[str] = []
    lines.append(
        "IFBench — play-by-play from **evaluation** JSONL (no orchestration trace in repo)\n"
        "(Canonical: ifbench_eval_results_playbyplay.txt)\n"
        "================================================================================\n\n"
        f"Source: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}\n"
        "Searched `results/semantic_metrics_logs/trace_*.jsonl` for IFBench prompt prefixes: **no matches**.\n"
        "Below: verbatim first rows (prompt + response + follow flags). For CEO/worker play-by-plays, log\n"
        "or import orchestration for IFBench the same way as StrategyQA.\n\n"
    )
    with p.open(encoding="utf-8", errors="replace") as f:
        for k, line in enumerate(f):
            if k >= 3:
                break
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            pr = o.get("prompt") or o.get("instruction") or ""
            resp = o.get("response") or o.get("model_response") or ""
            lines.append("-" * 88 + f"\nRow index {k}\n")
            lines.append("follow_all_instructions: " + str(o.get("follow_all_instructions")) + "\n")
            lines.append("instruction_id_list: " + str(o.get("instruction_id_list")) + "\n\n")
            lines.append("--- PROMPT (verbatim) ---\n")
            lines.append(str(pr) + "\n\n")
            lines.append("--- MODEL RESPONSE (verbatim) ---\n")
            lines.append(str(resp)[:15000] + ("…\n" if len(str(resp)) > 15000 else "\n"))
    return "".join(lines)


def write_other_benchmarks_index() -> str:
    return (
        "Other benchmarks — semantic metrics / traces index (read-only, for A/B curation)\n"
        "================================================================================\n\n"
        "Use this when a benchmark is not (yet) wired to per-question `semantic_metrics_logs` traces\n"
        "in the same way as StrategyQA or GSM8K.\n\n"
        "IFBench (instruction following)\n"
        "--------------------------------\n"
        f"  Code: bench/benchmarking_ifbench.py\n"
        f"  Local eval JSONL: bench/IFBench/eval/hashiru_20260319_*/ifbench_input_subset.jsonl\n"
        f"  Prompt-matched orchestration A/B (no `benchmark_name: ifbench`): see "
        f"ifbench_ablation_sd_se_vs_off_playbyplay.txt (regenerate from this script). "
        f"Exports are not tagged `ifbench` in {LOGS}; matching is by prompt substring on `user_turn_excerpt`.\n\n"
        "ToolBench\n"
        "---------\n"
        f"  Code: bench/benchmarking_toolbench.py, ToolBench/ server.\n"
        f"  README: bench/README_TOOLBENCH.md\n"
        f"  No ifbench-style trace shard in {LOGS} for ToolBench in this repo snapshot.\n\n"
        "Tau-2 (τ²) agent simulations\n"
        "-----------------------------\n"
        f"  Artifacts under results/: tau2_retail_*, tau2_airline_*, tau2_telecom_*_summary*.jsonl\n"
        f"  (17 files in this tree). These are **aggregate** summary rows, not per-question\n"
        f"  orchestration jsonl. Example keys: domain, num_tasks, num_trials, metrics, simulations_count.\n"
        f"  For AskAgent/AskMultipleAgents + semantic_entropy/semantic_density play-by-plays, add a trace\n"
        f"  export from the τ² driver if you need line-level A/B; summaries alone are not enough.\n\n"
        "Paper review (accept/reject)\n"
        "----------------------------\n"
        f"  Results: bench/results/paper_review_benchmark_*.jsonl\n"
        f"  Compare runs with `semantic_metrics_called: true` vs `false` in the JSONL (see inventory). "
        f"There is no matching `paper_review` `benchmark_name` in semantic_metrics_logs/ in the inventory; "
        f"item-level A/B for the same `paper_id` would require two result rows + optional trace export.\n\n"
        "JailbreakBench / TruthfulQA / MMLU\n"
        "----------------------------------\n"
        f"  See: jailbreakbench_ablation_semantic_playbyplay.txt, truthfulqa_ablation_semantic_playbyplay.txt,\n"
        f"  mmlu_law_ablation_semantic_playbyplay.txt, gsm8k_ablation_semantic_playbyplay.txt, and\n"
        f"  strategyqa_ablation_semantic_helped_playbyplay.txt (full four-way ablation).\n"
        f"  Cross-trace question match (by text): _trace_question_text_matcher.py; example TSV:\n"
        f"  strategyqa_trace_text_overlap_20260406_vs_200844.txt\n"
        f"  Curated wrong+wrong (SD/SE vs off): confidently_wrong_sd_se_ablation_examples.txt\n"
        f"  Text-matched (question string): truthfulqa_text_matched_runs_playbyplay.txt,\n"
        f"  mmlu_pro_text_matched_runs_playbyplay.txt, ifbench_eval_results_playbyplay.txt,\n"
        f"  ifbench_ablation_sd_se_vs_off_playbyplay.txt\n"
    )


def main() -> None:
    (EVAL / "gsm8k_ablation_semantic_playbyplay.txt").write_text(
        generate_gsm8k(), encoding="utf-8"
    )
    (EVAL / "truthfulqa_ablation_semantic_playbyplay.txt").write_text(
        generate_truthful(), encoding="utf-8"
    )
    (EVAL / "truthfulqa_ablation_sd_se_vs_off_playbyplay.txt").write_text(
        generate_truthful_sd_se_pair(), encoding="utf-8"
    )
    (EVAL / "confidently_wrong_sd_se_ablation_examples.txt").write_text(
        generate_confidently_wrong_playbyplay(), encoding="utf-8"
    )
    (EVAL / "mmlu_pro_ablation_869_playbyplay.txt").write_text(
        generate_mmlu_pro_869_playbyplay(), encoding="utf-8"
    )
    (EVAL / "mmlu_law_ablation_semantic_playbyplay.txt").write_text(
        generate_mmlu_law(), encoding="utf-8"
    )
    (EVAL / "jailbreakbench_ablation_semantic_playbyplay.txt").write_text(
        generate_jailbreak(), encoding="utf-8"
    )
    (EVAL / "ifbench_toolbench_tau2_paperreview_semantic_artifacts_index.txt").write_text(
        write_other_benchmarks_index(), encoding="utf-8"
    )
    (EVAL / "truthfulqa_text_matched_runs_playbyplay.txt").write_text(
        generate_truthful_text_matched(), encoding="utf-8"
    )
    (EVAL / "mmlu_pro_text_matched_runs_playbyplay.txt").write_text(
        generate_mmlu_text_matched_870(), encoding="utf-8"
    )
    (EVAL / "ifbench_eval_results_playbyplay.txt").write_text(
        generate_ifbench_eval_playbyplay(), encoding="utf-8"
    )
    (EVAL / "ifbench_ablation_sd_se_vs_off_playbyplay.txt").write_text(
        generate_ifbench_sd_se_vs_off(), encoding="utf-8"
    )
    (EVAL / "strategyqa_both_vs_both_reprompt_diff_playbyplay.txt").write_text(
        generate_strategyqa_both_vs_both_reprompt_diff(), encoding="utf-8"
    )
    print(
        "Wrote:\n"
        f"  {EVAL / 'gsm8k_ablation_semantic_playbyplay.txt'}\n"
        f"  {EVAL / 'truthfulqa_ablation_semantic_playbyplay.txt'}\n"
        f"  {EVAL / 'truthfulqa_ablation_sd_se_vs_off_playbyplay.txt'}\n"
        f"  {EVAL / 'confidently_wrong_sd_se_ablation_examples.txt'}\n"
        f"  {EVAL / 'mmlu_pro_ablation_869_playbyplay.txt'}\n"
        f"  {EVAL / 'mmlu_law_ablation_semantic_playbyplay.txt'}\n"
        f"  {EVAL / 'jailbreakbench_ablation_semantic_playbyplay.txt'}\n"
        f"  {EVAL / 'ifbench_toolbench_tau2_paperreview_semantic_artifacts_index.txt'}\n"
        f"  {EVAL / 'truthfulqa_text_matched_runs_playbyplay.txt'}\n"
        f"  {EVAL / 'mmlu_pro_text_matched_runs_playbyplay.txt'}\n"
        f"  {EVAL / 'ifbench_eval_results_playbyplay.txt'}\n"
        f"  {EVAL / 'ifbench_ablation_sd_se_vs_off_playbyplay.txt'}\n"
        f"  {EVAL / 'strategyqa_both_vs_both_reprompt_diff_playbyplay.txt'}"
    )


if __name__ == "__main__":
    main()
