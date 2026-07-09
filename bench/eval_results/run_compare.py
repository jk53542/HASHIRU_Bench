import json
from pathlib import Path
from statistics import mean
from collections import defaultdict

B = Path(__file__).resolve().parents[1]
L = B / "results/semantic_metrics_logs"
OUT = Path(__file__).resolve().parent / "_compare_out.txt"


def load(p):
    t = p.read_text()
    r = []
    b = ""
    d = 0
    for c in t:
        b += c
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
        if d == 0 and b.strip():
            r.append(json.loads(b))
            b = ""
    return r


def trace_stats(p):
    pq = defaultdict(lambda: {"wa": 0, "rep": 0, "con": 0, "multi": 0, "create": 0})
    for line in p.read_text().splitlines():
        o = json.loads(line)
        qi = o.get("question_index")
        if qi is None:
            continue
        q = pq[int(qi)]
        if o.get("event") in ("worker_answer", "worker_answer_multi"):
            q["wa"] += 1
            if o.get("worker_reprompted_after_semantic_check"):
                q["rep"] += 1
            if o.get("semantic_quality_concern"):
                q["con"] += 1
        if o.get("event") == "ceo_tool_finished" and o.get("tool") == "AskMultipleAgents":
            q["multi"] = 1
        if o.get("event") == "ceo_tool_finished" and o.get("tool") in (
            "AgentCreator",
            "CreateAgent",
        ):
            q["create"] += 1
    return pq


pairs = [
    (
        "musique",
        "exact_match",
        L / "trace_20260611_065038_368770.jsonl",
        B / "musique_results/musique_benchmark_20260611_065043.jsonl",
        L / "trace_20260612_075301_1187170_HASHIRU.jsonl",
        B / "musique_results/musique_benchmark_20260612_075703.jsonl",
    ),
    (
        "hotpotqa",
        "exact_match",
        L / "trace_20260611_092516_552311.jsonl",
        B / "hotpotqa_results/hotpotqa_benchmark_20260611_093915.jsonl",
        L / "trace_20260612_091718_1328721_HASHIRU.jsonl",
        B / "hotpotqa_results/hotpotqa_benchmark_20260612_091733.jsonl",
    ),
]
lines = []
for name, metric, ht, hr, rt, rr in pairs:
    hs = load(hr)
    rs = load(rr)
    htq = trace_stats(ht)
    rtq = trace_stats(rt)
    ha = sum(bool(x.get(metric)) for x in hs) / len(hs)
    ra = sum(bool(x.get(metric)) for x in rs) / len(rs)
    ht_mean = mean(float(x.get("time_elapsed", 0)) for x in hs)
    rt_mean = mean(float(x.get("time_elapsed", 0)) for x in rs)
    hb = {int(x["question_num"]): x for x in hs}
    rb = {int(x["question_num"]): x for x in rs}
    hf = rf = both = neither = 0
    for qi in sorted(set(hb) & set(rb)):
        h = bool(hb[qi].get(metric))
        r = bool(rb[qi].get(metric))
        if h and r:
            both += 1
        elif h and not r:
            hf += 1
        elif r and not h:
            rf += 1
        else:
            neither += 1
    lines.append(
        f"{name}|HASSUM acc={ha:.3f}|HASHIRU acc={ra:.3f}|dacc={ra-ha:+.3f}|"
        f"Htime={ht_mean:.1f}|Rtime={rt_mean:.1f}|dtime={rt_mean-ht_mean:+.1f}|"
        f"Hrep={sum(q['rep'] for q in htq.values())}|Hworkers={sum(q['wa'] for q in htq.values())}|"
        f"Rworkers={sum(q['wa'] for q in rtq.values())}|Hmulti={sum(q['multi'] for q in htq.values())}|"
        f"Rmulti={sum(q['multi'] for q in rtq.values())}|Hcreate={sum(q['create'] for q in htq.values())}|"
        f"Rcreate={sum(q['create'] for q in rtq.values())}|Honly={hf}|Ronly={rf}|both={both}|neither={neither}"
    )
OUT.write_text("\n".join(lines))
