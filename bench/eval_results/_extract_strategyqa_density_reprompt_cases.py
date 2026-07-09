#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TRACE_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "semantic_metrics_logs"
    / "trace_20260415_000218_3275869.jsonl"
)
OUT_PATH = Path(__file__).resolve().parent / "strategyqa_density_only_reprompt_cases_playbyplay.txt"
TARGET_QIDS = ["40", "42", "45"]


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
    buf.append(f"(tool {tool!r}: prompts/responses not expanded by this extractor)\n")


def build_case(trace_name: str, qid: str, events: list[dict]) -> str:
    finishes = [
        e
        for e in events
        if e.get("event") == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]
    qtext = next(
        (
            e.get("question_text")
            for e in events
            if isinstance(e.get("question_text"), str) and e.get("question_text").strip()
        ),
        "",
    )
    ceo_finals = [e for e in events if e.get("event") == "ceo_final_answer"]
    ceo_finals.sort(key=lambda x: float(x.get("ts") or 0.0))

    n = len(finishes)
    rep = max(0, n - 1)
    density_lows = [
        e
        for e in finishes
        if isinstance(e.get("semantic_density"), (int, float))
        and isinstance(e.get("semantic_density_threshold"), (int, float))
        and e.get("semantic_density") < e.get("semantic_density_threshold")
    ]
    reprompt_rows = [e for e in finishes if e.get("worker_reprompted_after_semantic_check") is True]

    out: list[str] = []
    out.append("=" * 92 + "\n")
    out.append(f"QUESTION ID: {qid}\n")
    out.append(f"Worker tool rounds (AskAgent / AskMultipleAgents): {n}\n")
    out.append(f"CEO reprompts after first worker round: {rep}\n")
    out.append(f"Density-below-threshold worker rounds: {len(density_lows)}\n")
    out.append(f"Rounds marked worker_reprompted_after_semantic_check=true: {len(reprompt_rows)}\n")
    out.append(f"Trace file: {trace_name}\n")
    out.append("\n--- ORIGINAL BENCHMARK QUESTION (question_text) ---\n")
    out.append((qtext or "(missing question_text)") + "\n")
    out.append("\n--- HIGH-LEVEL SUMMARY ---\n")
    out.append(
        f"This density-only ablation case had {n} worker tool completion(s). "
        f"The CEO ran {rep} follow-up worker cycle(s) after the first attempt.\n"
    )
    seq = [str(e.get("tool")) for e in finishes]
    out.append(f"Sequence of tools: {' → '.join(seq) if seq else '(none)'}\n")
    if density_lows:
        vals = ", ".join(f"{float(e.get('semantic_density')):.4f}" for e in density_lows)
        out.append(
            f"Density crossed threshold (< {density_lows[0].get('semantic_density_threshold')!r}) on "
            f"{len(density_lows)} round(s): {vals}\n"
        )
    if reprompt_rows:
        out.append(
            "Reprompt-marked rounds (worker_reprompted_after_semantic_check=true): "
            f"{len(reprompt_rows)}\n"
        )

    out.append("\n--- FULL PLAY-BY-PLAY (exact prompts and responses) ---\n")
    for i, e in enumerate(finishes):
        emit_round(out, i, e)

    out.append("\n--- CEO FINAL ANSWER (exact) ---\n")
    if ceo_finals:
        out.append(str(ceo_finals[-1].get("ceo_final_answer") or ""))
        out.append("\n")
    else:
        out.append("(no ceo_final_answer row found for this question_id)\n")
    return "".join(out)


def main() -> None:
    rows: list[dict] = []
    with TRACE_PATH.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("question_id")) in TARGET_QIDS:
                rows.append(obj)

    by_q: dict[str, list[dict]] = defaultdict(list)
    for e in rows:
        by_q[str(e.get("question_id"))].append(e)
    for q in by_q:
        by_q[q].sort(key=lambda x: float(x.get("ts") or 0.0))

    parts: list[str] = []
    parts.append(
        "StrategyQA density-only run — reprompted cases (verbatim CEO↔worker play-by-play)\n"
    )
    parts.append(f"Trace: {TRACE_PATH.name}\n")
    parts.append("Cases included: question_id 40, 42, 45\n")
    parts.append("=" * 92 + "\n\n")
    for qid in TARGET_QIDS:
        parts.append(build_case(TRACE_PATH.name, qid, by_q.get(qid, [])))
        parts.append("\n")

    OUT_PATH.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
