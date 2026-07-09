#!/usr/bin/env python3
"""Compare HASSUM (metrics on) vs HASHIRU (metrics off) benchmark runs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

BENCH = Path(__file__).resolve().parents[1]
LOGS = BENCH / "results/semantic_metrics_logs"

RUNS = {
    "musique": {
        "hassum": {
            "trace": LOGS / "trace_20260611_065038_368770.jsonl",
            "results": BENCH / "musique_results/musique_benchmark_20260611_065043.jsonl",
            "metric": "exact_match",
        },
        "hashiru": {
            "trace": LOGS / "trace_20260612_075301_1187170_HASHIRU.jsonl",
            "results": BENCH / "musique_results/musique_benchmark_20260612_075703.jsonl",
            "metric": "exact_match",
        },
    },
    "hotpotqa": {
        "hassum": {
            "trace": LOGS / "trace_20260611_092516_552311.jsonl",
            "results": BENCH / "hotpotqa_results/hotpotqa_benchmark_20260611_093915.jsonl",
            "metric": "exact_match",
        },
        "hashiru": {
            "trace": LOGS / "trace_20260612_091718_1328721_HASHIRU.jsonl",
            "results": BENCH / "hotpotqa_results/hotpotqa_benchmark_20260612_091733.jsonl",
            "metric": "exact_match",
        },
    },
    "gpqa": {
        "hassum": {
            "trace": LOGS / "trace_20260611_124314_1055898.jsonl",
            "results": BENCH / "gpqa_results/gpqa_gpqa_diamond_benchmark_20260611_131116.jsonl",
            "metric": "is_correct",
        },
        "hashiru": None,  # no completed HASHIRU GPQA run yet
    },
}


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    recs: list[dict] = []
    buf = ""
    depth = 0
    for ch in text:
        buf += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and buf.strip():
                recs.append(json.loads(buf))
                buf = ""
    return recs


def parse_trace(path: Path) -> dict[int, dict]:
    per_q: dict[int, dict] = defaultdict(
        lambda: {
            "worker_answers": [],
            "ask_agent_tools": 0,
            "ask_multi_tools": 0,
            "create_agent": 0,
            "reprompts": 0,
            "concerns": 0,
            "entropy_vals": [],
            "density_vals": [],
            "metrics_active": False,
            "agents": [],
        }
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        qi = o.get("question_index")
        if qi is None:
            continue
        q = per_q[int(qi)]
        ev = o.get("event")
        if ev in ("worker_answer", "worker_answer_multi"):
            q["worker_answers"].append(o)
            if o.get("agent_name"):
                q["agents"].append(o["agent_name"])
            if o.get("worker_reprompted_after_semantic_check"):
                q["reprompts"] += 1
            if o.get("semantic_quality_concern"):
                q["concerns"] += 1
            ent = o.get("semantic_entropy")
            den = o.get("semantic_density")
            if ent is not None:
                q["entropy_vals"].append(float(ent))
                q["metrics_active"] = True
            if den is not None:
                q["density_vals"].append(float(den))
                q["metrics_active"] = True
        elif ev == "ceo_tool_finished":
            tool = o.get("tool")
            if tool == "AskAgent":
                q["ask_agent_tools"] += 1
            elif tool == "AskMultipleAgents":
                q["ask_multi_tools"] += 1
            elif tool in ("AgentCreator", "CreateAgent"):
                q["create_agent"] += 1
        elif ev == "ceo_create_agent":
            q["create_agent"] += 1

    for q in per_q.values():
        q["n_workers"] = len(q["worker_answers"])
        q["unique_agents"] = len(set(q["agents"]))
        q["had_multi"] = q["ask_multi_tools"] > 0 or any(
            w.get("worker_routing") == "AskMultipleAgents" for w in q["worker_answers"]
        )
    return dict(per_q)


def summarize_results(recs: list[dict], metric: str) -> dict:
    times = [float(r.get("time_elapsed", 0)) for r in recs]
    correct = [bool(r.get(metric)) for r in recs]
    return {
        "n": len(recs),
        "accuracy": sum(correct) / len(correct) if correct else 0,
        "correct": sum(correct),
        "mean_time": mean(times) if times else 0,
        "median_time": median(times) if times else 0,
        "total_time": sum(times),
        "mandate_violations": sum(1 for r in recs if r.get("worker_mandate_violation")),
        "by_q": {int(r["question_num"]): r for r in recs},
    }


def summarize_trace(trace_q: dict[int, dict]) -> dict:
    rows = list(trace_q.values())
    if not rows:
        return {}
    return {
        "questions": len(rows),
        "metrics_active": any(r["metrics_active"] for r in rows),
        "total_reprompts": sum(r["reprompts"] for r in rows),
        "questions_with_reprompt": sum(1 for r in rows if r["reprompts"] > 0),
        "total_concerns": sum(r["concerns"] for r in rows),
        "questions_with_concern": sum(1 for r in rows if r["concerns"] > 0),
        "mean_workers": mean(r["n_workers"] for r in rows),
        "mean_ask_agent": mean(r["ask_agent_tools"] for r in rows),
        "questions_multi": sum(1 for r in rows if r["had_multi"]),
        "total_create": sum(r["create_agent"] for r in rows),
        "mean_unique_agents": mean(r["unique_agents"] for r in rows),
        "mean_entropy": mean(
            mean(r["entropy_vals"]) for r in rows if r["entropy_vals"]
        )
        if any(r["entropy_vals"] for r in rows)
        else None,
        "mean_density": mean(
            mean(r["density_vals"]) for r in rows if r["density_vals"]
        )
        if any(r["density_vals"] for r in rows)
        else None,
        "per_q": trace_q,
    }


def paired_flip_analysis(
    bench: str, hassum_res: dict, hashiru_res: dict, metric: str
) -> dict:
    """Questions where only one mode got it right."""
    h = hassum_res["by_q"]
    r = hashiru_res["by_q"]
    common = sorted(set(h) & set(r))
    hassum_only = []
    hashiru_only = []
    both_right = both_wrong = 0
    for qi in common:
        hc = bool(h[qi].get(metric))
        rc = bool(r[qi].get(metric))
        if hc and rc:
            both_right += 1
        elif not hc and not rc:
            both_wrong += 1
        elif hc and not rc:
            hassum_only.append(qi)
        else:
            hashiru_only.append(qi)
    return {
        "common": len(common),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "hassum_only_correct": len(hassum_only),
        "hashiru_only_correct": len(hashiru_only),
        "hassum_only_qs": hassum_only[:10],
        "hashiru_only_qs": hashiru_only[:10],
    }


def main() -> None:
    print("HASSUM vs HASHIRU Performance Comparison")
    print("=" * 70)
    all_paired = {}

    for bench, cfg in RUNS.items():
        print(f"\n### {bench.upper()} ###")
        hcfg = cfg["hassum"]
        rcfg = cfg.get("hashiru")
        if not hcfg["trace"].exists():
            print("  Missing HASSUM trace")
            continue
        h_res = summarize_results(load_jsonl(hcfg["results"]), hcfg["metric"])
        h_trace = summarize_trace(parse_trace(hcfg["trace"]))
        print(f"\nHASSUM (entropy+density ON):")
        print(f"  Accuracy: {h_res['correct']}/{h_res['n']} = {h_res['accuracy']:.1%}")
        print(f"  Time: mean {h_res['mean_time']:.0f}s, median {h_res['median_time']:.0f}s, total {h_res['total_time']/3600:.1f}h")
        print(f"  Reprompts: {h_trace.get('total_reprompts',0)} ({h_trace.get('questions_with_reprompt',0)} questions)")
        print(f"  Concern flags: {h_trace.get('total_concerns',0)} ({h_trace.get('questions_with_concern',0)} questions)")
        print(f"  Workers/q: {h_trace.get('mean_workers',0):.2f}, AskAgent tools/q: {h_trace.get('mean_ask_agent',0):.2f}")
        print(f"  Multi-agent questions: {h_trace.get('questions_multi',0)}, Agent creates: {h_trace.get('total_create',0)}")
        if h_trace.get("mean_entropy") is not None:
            print(f"  Mean entropy: {h_trace['mean_entropy']:.3f}, density: {h_trace['mean_density']:.3f}")

        if rcfg is None or not rcfg["trace"].exists():
            print("\nHASHIRU: no completed run to compare")
            continue
        r_res = summarize_results(load_jsonl(rcfg["results"]), rcfg["metric"])
        r_trace = summarize_trace(parse_trace(rcfg["trace"]))
        print(f"\nHASHIRU (entropy+density OFF):")
        print(f"  Accuracy: {r_res['correct']}/{r_res['n']} = {r_res['accuracy']:.1%}")
        print(f"  Time: mean {r_res['mean_time']:.0f}s, median {r_res['median_time']:.0f}s, total {r_res['total_time']/3600:.1f}h")
        print(f"  Reprompts: {r_trace.get('total_reprompts',0)} (expected 0)")
        print(f"  Workers/q: {r_trace.get('mean_workers',0):.2f}, AskAgent tools/q: {r_trace.get('mean_ask_agent',0):.2f}")
        print(f"  Multi-agent questions: {r_trace.get('questions_multi',0)}, Agent creates: {r_trace.get('total_create',0)}")
        metrics_null = r_trace.get("mean_entropy") is None and not r_trace.get("metrics_active")
        print(f"  Metrics inactive in trace: {metrics_null}")

        delta_acc = h_res["accuracy"] - r_res["accuracy"]
        delta_time = h_res["mean_time"] - r_res["mean_time"]
        print(f"\nDELTA (HASSUM - HASHIRU):")
        print(f"  Accuracy: {delta_acc:+.1%} ({h_res['correct'] - r_res['correct']:+d} questions)")
        print(f"  Mean time/question: {delta_time:+.0f}s ({delta_time/r_res['mean_time']*100:+.1f}%)" if r_res['mean_time'] else "")
        print(f"  Extra worker completions: {(h_trace.get('mean_workers',0)-r_trace.get('mean_workers',0)):+.2f}/q")
        print(f"  Extra reprompts (HASSUM only): {h_trace.get('total_reprompts',0)}")

        paired = paired_flip_analysis(bench, h_res, r_res, hcfg["metric"])
        all_paired[bench] = paired
        print(f"\nPaired (same question_num, n={paired['common']}):")
        print(f"  Both correct: {paired['both_right']}, Both wrong: {paired['both_wrong']}")
        print(f"  HASSUM only correct: {paired['hassum_only_correct']}")
        print(f"  HASHIRU only correct: {paired['hashiru_only_correct']}")

    print("\n" + "=" * 70)
    print("SUMMARY")


if __name__ == "__main__":
    main()
