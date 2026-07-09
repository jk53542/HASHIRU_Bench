#!/usr/bin/env python3
"""Analyze HASSUM semantic metrics impact from traces + benchmark JSONL."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

BENCH = Path(__file__).resolve().parents[1]
TRACES = {
    "musique": BENCH / "results/semantic_metrics_logs/trace_20260611_065038_368770.jsonl",
    "hotpotqa": BENCH / "results/semantic_metrics_logs/trace_20260611_092516_552311.jsonl",
    "gpqa": BENCH / "results/semantic_metrics_logs/trace_20260611_124314_1055898.jsonl",
}
RESULTS = {
    "musique": BENCH / "musique_results/musique_benchmark_20260611_065043.jsonl",
    "hotpotqa": BENCH / "hotpotqa_results/hotpotqa_benchmark_20260611_093915.jsonl",
    "gpqa": BENCH / "gpqa_results/gpqa_gpqa_diamond_benchmark_20260611_131116.jsonl",
}


def load_benchmark_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    buf = ""
    depth = 0
    for ch in text:
        buf += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and buf.strip():
                records.append(json.loads(buf))
                buf = ""
    return records


def parse_trace(path: Path) -> dict:
    per_q: dict[int, dict] = defaultdict(lambda: {
        "worker_answers": [],
        "ask_agent": 0,
        "ask_multiple": 0,
        "create_agent": 0,
        "agents_used": [],
        "reprompts": 0,
        "concerns": 0,
        "entropy_vals": [],
        "density_vals": [],
        "final_answer": None,
        "benchmark": None,
    })
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        qi = o.get("question_index")
        if qi is None:
            continue
        q = per_q[int(qi)]
        if o.get("benchmark_name"):
            q["benchmark"] = o["benchmark_name"]
        ev = o.get("event")
        if ev == "worker_answer":
            q["worker_answers"].append(o)
            q["ask_agent"] += 1
            if o.get("agent_name"):
                q["agents_used"].append(o["agent_name"])
            if o.get("worker_reprompted_after_semantic_check"):
                q["reprompts"] += 1
            if o.get("semantic_quality_concern"):
                q["concerns"] += 1
            if o.get("semantic_entropy") is not None:
                q["entropy_vals"].append(float(o["semantic_entropy"]))
            if o.get("semantic_density") is not None:
                q["density_vals"].append(float(o["semantic_density"]))
        elif ev == "worker_answer_multi":
            q["worker_answers"].append(o)
            q["ask_multiple"] += 1
            if o.get("agent_name"):
                q["agents_used"].append(o["agent_name"])
            if o.get("worker_reprompted_after_semantic_check"):
                q["reprompts"] += 1
            if o.get("semantic_quality_concern"):
                q["concerns"] += 1
            if o.get("semantic_entropy") is not None:
                q["entropy_vals"].append(float(o["semantic_entropy"]))
            if o.get("semantic_density") is not None:
                q["density_vals"].append(float(o["semantic_density"]))
        elif ev == "ceo_tool_finished":
            tool = o.get("tool")
            if tool == "AskAgent":
                pass  # counted via worker_answer
            elif tool == "AskMultipleAgents":
                q["ask_multiple"] += 1
            elif tool in ("AgentCreator", "CreateAgent"):
                q["create_agent"] += 1
        elif ev == "ceo_create_agent":
            q["create_agent"] += 1
        elif ev == "ceo_final_answer":
            q["final_answer"] = o.get("ceo_final_answer")

    # dedupe ask_multiple from tool_finished + worker_answer_multi: use worker_answer count for multi subcalls
    for qi, q in per_q.items():
        multi_worker = sum(1 for w in q["worker_answers"] if w.get("worker_routing") == "AskMultipleAgents")
        q["multi_subcalls"] = multi_worker
        q["unique_agents"] = len(set(q["agents_used"]))
        q["agent_switches"] = max(0, q["unique_agents"] - 1)
    return dict(per_q)


def correctness_key(rec: dict, bench: str) -> bool:
    if bench == "gpqa":
        return bool(rec.get("is_correct"))
    if bench == "hotpotqa":
        return bool(rec.get("exact_match") or rec.get("is_correct"))
    return bool(rec.get("exact_match") or rec.get("is_correct"))


def correlate(bench: str, trace_q: dict, results: list[dict]) -> dict:
    by_num = {int(r.get("question_num", r.get("question_index", 0))): r for r in results}
    rows = []
    for qi in sorted(trace_q.keys()):
        if qi not in by_num:
            continue
        t = trace_q[qi]
        r = by_num[qi]
        correct = correctness_key(r, bench)
        rows.append({
            "qi": qi,
            "correct": correct,
            "reprompts": t["reprompts"],
            "concerns": t["concerns"],
            "ask_agent": t["ask_agent"],
            "multi_subcalls": t["multi_subcalls"],
            "create_agent": t["create_agent"],
            "unique_agents": t["unique_agents"],
            "agent_switches": t["agent_switches"],
            "n_workers": len(t["worker_answers"]),
            "entropy_mean": mean(t["entropy_vals"]) if t["entropy_vals"] else None,
            "density_mean": mean(t["density_vals"]) if t["density_vals"] else None,
            "had_concern": t["concerns"] > 0,
            "had_reprompt": t["reprompts"] > 0,
            "had_multi": t["multi_subcalls"] > 0,
            "had_create": t["create_agent"] > 0,
            "had_switch": t["unique_agents"] > 1,
        })

    def acc(subset):
        if not subset:
            return None
        return sum(1 for x in subset if x["correct"]) / len(subset)

    def summarize(flag_key):
        pos = [x for x in rows if x[flag_key]]
        neg = [x for x in rows if not x[flag_key]]
        return {
            "n": len(pos),
            "accuracy": acc(pos),
            "n_without": len(neg),
            "accuracy_without": acc(neg),
        }

    # reprompt outcome: first concern same-agent then reprompt
    reprompt_help = 0
    reprompt_hurt = 0
    reprompt_neutral = 0
    for qi, t in trace_q.items():
        if qi not in by_num:
            continue
        wa = sorted(
            [w for w in t["worker_answers"] if w.get("event") != "worker_answer_multi" or True],
            key=lambda w: (w.get("agent_name", ""), w.get("worker_invocation_index") or 1),
        )
        by_agent: dict[str, list] = defaultdict(list)
        for w in t["worker_answers"]:
            by_agent[w.get("agent_name", "?")].append(w)
        for agent, calls in by_agent.items():
            calls = sorted(calls, key=lambda w: w.get("worker_invocation_index") or 1)
            for i in range(1, len(calls)):
                prev, cur = calls[i - 1], calls[i]
                if prev.get("semantic_quality_concern") and cur.get("worker_reprompted_after_semantic_check"):
                    # can't know per-step correctness; mark question-level
                    if correctness_key(by_num[qi], bench):
                        reprompt_help += 1
                    else:
                        reprompt_hurt += 1
                    break

    return {
        "n_questions": len(rows),
        "accuracy": acc(rows),
        "rows": rows,
        "had_concern": summarize("had_concern"),
        "had_reprompt": summarize("had_reprompt"),
        "had_multi": summarize("had_multi"),
        "had_create": summarize("had_create"),
        "had_switch": summarize("had_switch"),
        "totals": {
            "reprompts": sum(x["reprompts"] for x in rows),
            "concerns": sum(x["concerns"] for x in rows),
            "multi_subcalls": sum(x["multi_subcalls"] for x in rows),
            "create_agent": sum(x["create_agent"] for x in rows),
            "mean_workers": mean(x["n_workers"] for x in rows) if rows else 0,
            "mean_entropy": mean(x["entropy_mean"] for x in rows if x["entropy_mean"] is not None) if rows else None,
            "mean_density": mean(x["density_mean"] for x in rows if x["density_mean"] is not None) if rows else None,
        },
        "reprompt_chain_questions": {"help_correct": reprompt_help, "still_wrong": reprompt_hurt},
    }


def print_bench(name: str, stats: dict) -> None:
    print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
    print(f"Questions matched: {stats['n_questions']}, accuracy: {stats['accuracy']:.1%}" if stats['accuracy'] is not None else "no data")
    t = stats["totals"]
    print(f"Total semantic reprompts: {t['reprompts']}, concern flags: {t['concerns']}")
    print(f"Multi-agent subcalls: {t['multi_subcalls']}, agent creates: {t['create_agent']}")
    print(f"Mean worker completions/q: {t['mean_workers']:.2f}, mean entropy: {t['mean_entropy']:.3f}, mean density: {t['mean_density']:.3f}" if t['mean_entropy'] else "")
    for key in ("had_concern", "had_reprompt", "had_multi", "had_create", "had_switch"):
        s = stats[key]
        if s["n"] == 0 and s["n_without"] == 0:
            continue
        a = s["accuracy"]
        b = s["accuracy_without"]
        a_s = f"{a:.1%}" if a is not None else "n/a"
        b_s = f"{b:.1%}" if b is not None else "n/a"
        print(f"  {key}: n={s['n']} acc={a_s} | without: n={s['n_without']} acc={b_s}")


def main():
    for bench in ("musique", "hotpotqa", "gpqa"):
        trace_path = TRACES[bench]
        res_path = RESULTS[bench]
        if not trace_path.exists():
            print(f"Missing trace {trace_path}")
            continue
        trace_q = parse_trace(trace_path)
        results = load_benchmark_jsonl(res_path)
        stats = correlate(bench, trace_q, results)
        print_bench(bench, stats)

    # Cross-bench: concern -> CEO action patterns
    print(f"\n{'='*60}\nCROSS-BENCH SEMANTIC BEHAVIOR\n{'='*60}")
    for bench in ("musique", "hotpotqa", "gpqa"):
        trace_q = parse_trace(TRACES[bench])
        results = load_benchmark_jsonl(RESULTS[bench])
        by_num = {int(r.get("question_num", 0)): r for r in results}
        after_concern_reprompt = 0
        after_concern_switch = 0
        after_concern_ignore = 0
        for qi, t in trace_q.items():
            if qi not in by_num:
                continue
            wa = t["worker_answers"]
            if not any(w.get("semantic_quality_concern") for w in wa):
                continue
            if t["reprompts"] > 0:
                after_concern_reprompt += 1
            elif t["unique_agents"] > 1:
                after_concern_switch += 1
            else:
                after_concern_ignore += 1
        n = after_concern_reprompt + after_concern_switch + after_concern_ignore
        if n:
            print(f"{bench}: after≥1 concern → reprompt same agent: {after_concern_reprompt}/{n}, switch agent: {after_concern_switch}/{n}, no follow-up worker: {after_concern_ignore}/{n}")


if __name__ == "__main__":
    main()
