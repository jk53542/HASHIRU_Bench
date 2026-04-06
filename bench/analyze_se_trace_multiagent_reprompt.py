#!/usr/bin/env python3
"""
Tables from SE/SD benchmark JSONL + traces: multi-agent (AskMultipleAgents) vs AskAgent-only,
and **CEO follow-up after a threshold-flagged worker finish** (trace-grounded, not timing heuristics).

Aligned rows use the same matching rules as analyze_se_trace_call_accuracy.py.

Why traces look repetitive (e.g. worker_answer → ceo_ask_agent → ceo_tool_finished)
------------------------------------------------------------------
Each ``AskAgent`` completion is logged in three places: the worker line, the tool’s
``ceo_ask_agent`` line (same metrics), and ``ceo_tool_finished`` (same metrics + tool
result ``message``). That is expected duplication, not three separate model calls.

``semantic_quality_concern`` / “crossed thresholds”
---------------------------------------------------
``reprompt_triggered`` in HASHIRU is true when **either** enabled branch trips:
entropy **>** threshold **or** density **<** threshold. A **tiny** entropy (e.g. ~1e-36)
with **concern true** usually means **density** alone failed (e.g. 0.59 < 0.8), not entropy.

The AskAgent tool **does not auto-re-prompt** the worker. When concern is true, the
``ceo_tool_finished`` ``message`` is the long form (“Agent replied, but semantic
entropy/density crossed thresholds…”). A **later** ``AskAgent`` / ``AskMultipleAgents``
finish with a **different** ``worker_prompt`` is a **new CEO tool call** (CEO-chosen
follow-up), which we count as “follow-up after threshold signal” when it comes **after**
a threshold-flagged finish in the same aligned benchmark row.

``worker_reprompted_after_semantic_check`` is tabulated separately; it is often false even
when the CEO does ask again, depending on orchestration state.

Usage:
  python HASHIRU_Bench/bench/analyze_se_trace_multiagent_reprompt.py TRACE_FILENAME.jsonl
  python ... --all
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import analyze_se_trace_call_accuracy as se

_THRESHOLD_MSG_NEEDLE = "crossed thresholds"


def ceo_finish_has_threshold_signal(e: dict) -> bool:
    """
    True when this ``ceo_tool_finished`` row indicates SE/SD gates fired for that tool
    completion (same signals the CEO sees in the tool ``message``).
    """
    if e.get("semantic_quality_concern") is True:
        return True
    msg = str(e.get("message") or "").lower()
    if _THRESHOLD_MSG_NEEDLE in msg:
        return True
    if e.get("tool") == "AskMultipleAgents":
        pa = e.get("per_agent_outputs") or []
        if isinstance(pa, list):
            for row in pa:
                if isinstance(row, dict) and row.get("semantic_quality_concern") is True:
                    return True
    return False


def row_has_ask_multiple(evs: list[dict]) -> bool:
    return any(e.get("tool") == "AskMultipleAgents" for e in evs)


def row_followup_after_threshold_signal(evs: list[dict]) -> bool:
    """
    Some worker ``ceo_tool_finished`` row was threshold-flagged, and a **later**
    matched worker finish exists (another AskAgent or AskMultipleAgents in this row).
    """
    ordered = sorted(evs, key=lambda x: float(x.get("ts") or 0.0))
    for i in range(len(ordered)):
        if not ceo_finish_has_threshold_signal(ordered[i]):
            continue
        if i + 1 < len(ordered):
            return True
    return False


def row_reprompt_trace_flag(evs: list[dict]) -> bool:
    return any(e.get("worker_reprompted_after_semantic_check") is True for e in evs)


def table_two_way(
    rows: dict[str, dict[str, int]],
    title: str,
    row_labels: list[str],
) -> str:
    lines = [
        f"### {title}",
        "",
        "| Category | Correct | Incorrect | Total |",
        "|---|---:|---:|---:|",
    ]
    for key in row_labels:
        c = rows.get(key, {"correct": 0, "incorrect": 0})
        tot = c.get("correct", 0) + c.get("incorrect", 0)
        lines.append(
            f"| {key} | {c.get('correct', 0)} | {c.get('incorrect', 0)} | {tot} |"
        )
    tc = sum(rows[k].get("correct", 0) for k in row_labels if k in rows)
    ti = sum(rows[k].get("incorrect", 0) for k in row_labels if k in rows)
    tt = tc + ti
    lines.append(f"| **All aligned** | **{tc}** | **{ti}** | **{tt}** |")
    return "\n".join(lines)


def table_cross_multi_followup(
    cross: dict[tuple[bool, bool], dict[str, int]],
    title: str,
) -> str:
    """Keys: (used_ask_multiple, followup_after_threshold)."""
    lines = [
        f"### {title}",
        "",
        "| AskMultipleAgents used? | Follow-up worker after threshold signal? | Correct | Incorrect | Total |",
        "|---|---|---:|---:|---:|",
    ]
    order = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    for mu, fu in order:
        c = cross.get((mu, fu), {"correct": 0, "incorrect": 0})
        tot = c.get("correct", 0) + c.get("incorrect", 0)
        lines.append(
            f"| {'Yes' if mu else 'No'} | {'Yes' if fu else 'No'} | "
            f"{c.get('correct', 0)} | {c.get('incorrect', 0)} | {tot} |"
        )
    tc = sum(cross.get(k, {}).get("correct", 0) for k in order)
    ti = sum(cross.get(k, {}).get("incorrect", 0) for k in order)
    lines.append(
        f"| **All aligned** |  | **{tc}** | **{ti}** | **{tc + ti}** |"
    )
    return "\n".join(lines)


def run_benchmark_tables(
    name: str,
    results_path: Path,
    trace_path: Path,
    correct_key: str,
    mode: str,
    trace_bench_slug: str,
) -> tuple[str, int, int]:
    results = se.load_benchmark_results(results_path)
    events = se.load_trace_events(trace_path)

    multi_grid: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    follow_grid: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    flag_grid: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    cross: dict[tuple[bool, bool], dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0}
    )
    no_match = 0

    for i, r in enumerate(results):
        evs, matched = se.get_matched_worker_events(
            events,
            r,
            mode=mode,
            trace_bench_slug=trace_bench_slug,
            row_index_1based=i + 1,
        )
        if not matched:
            no_match += 1
            continue
        ok = se._is_correct_result(r, correct_key)
        bucket = "correct" if ok else "incorrect"

        has_multi = row_has_ask_multiple(evs)
        has_follow = row_followup_after_threshold_signal(evs)

        if has_multi:
            multi_grid["Used AskMultipleAgents (≥1)"][bucket] += 1
        else:
            multi_grid["AskAgent only (no AskMultipleAgents)"][bucket] += 1

        if has_follow:
            follow_grid[
                "CEO follow-up: threshold-flagged finish, then another worker finish"
            ][bucket] += 1
        else:
            follow_grid["No CEO worker follow-up after threshold signal"][bucket] += 1

        if row_reprompt_trace_flag(evs):
            flag_grid["worker_reprompted_after_semantic_check true"][bucket] += 1
        else:
            flag_grid["worker_reprompted_after_semantic_check false / missing"][bucket] += 1

        cross[(has_multi, has_follow)][bucket] += 1

    aligned = len(results) - no_match
    header = f"## {name}\n\n*Trace-aligned rows: **{aligned}** of **{len(results)}**"
    if no_match:
        header += f"; **{no_match}** row(s) had no matching worker tools.*\n"
    else:
        header += ".*\n"

    m_md = table_two_way(
        dict(multi_grid),
        f"{name} — multi-agent routing",
        ["AskAgent only (no AskMultipleAgents)", "Used AskMultipleAgents (≥1)"],
    )
    f_md = table_two_way(
        dict(follow_grid),
        f"{name} — CEO follow-up after threshold (trace-based)",
        [
            "No CEO worker follow-up after threshold signal",
            "CEO follow-up: threshold-flagged finish, then another worker finish",
        ],
    )
    flag_md = table_two_way(
        dict(flag_grid),
        f"{name} — trace reprompt flag",
        [
            "worker_reprompted_after_semantic_check false / missing",
            "worker_reprompted_after_semantic_check true",
        ],
    )
    x_md = table_cross_multi_followup(
        dict(cross),
        f"{name} — cross: AskMultipleAgents × threshold follow-up",
    )
    return (
        header + "\n\n" + m_md + "\n\n" + f_md + "\n\n" + x_md + "\n\n" + flag_md,
        len(results),
        no_match,
    )


def main() -> None:
    bench = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Multi-agent / threshold follow-up tables from benchmark JSONL + trace."
    )
    se.add_trace_cli_arguments(parser)
    args = parser.parse_args()
    jobs = se.jobs_from_args(bench, args)

    sections: list[str] = []
    for name, rp, tp, ck, mode, slug in jobs:
        if not rp.is_file():
            sections.append(f"## {name}\n\n*Missing results: {rp}*\n")
            continue
        if not tp.is_file():
            sections.append(f"## {name}\n\n*Missing trace: {tp}*\n")
            continue
        md, _, _ = run_benchmark_tables(name, rp, tp, ck, mode, slug)
        sections.append(md)

    out = "\n\n".join(sections)
    print(out)
    out_path = bench / "results" / "se_multiagent_reprompt_tables.md"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out + "\n", encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    main()
