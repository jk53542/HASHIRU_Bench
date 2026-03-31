"""One-off stats for agent analysis; delete after use."""
import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent


def load_jsonl(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def main():
    jb = load_jsonl(BASE / "results" / "jailbreakbench_benchmark_20260328_052548_modif.jsonl")
    sq = []
    p = BASE / "strategyqa_results" / "strategyqa_benchmark_20260330_023918_modif.jsonl"
    raw = p.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \t\n\r":
            idx += 1
        if idx >= len(raw):
            break
        obj, end = dec.raw_decode(raw, idx)
        sq.append(obj)
        idx = end

    tq = []
    p2 = BASE / "truthful_qa_results" / "truthful_qa_benchmark_20260330_203313_modif.jsonl"
    raw2 = p2.read_text(encoding="utf-8")
    idx = 0
    while idx < len(raw2):
        while idx < len(raw2) and raw2[idx] in " \t\n\r":
            idx += 1
        if idx >= len(raw2):
            break
        obj, end = dec.raw_decode(raw2, idx)
        tq.append(obj)
        idx = end

    print("JailbreakBench samples:", len(jb))
    if jb:
        c = Counter(r["is_correct"] for r in jb)
        print("  is_correct:", dict(c), f"rate={c[True]/len(jb):.4f}")
        fail = [r for r in jb if not r["is_correct"]]
        print("  sample false categories:", Counter(r.get("category") for r in fail).most_common(8))

    print("StrategyQA samples:", len(sq))
    if sq:
        c = Counter(r["is_correct"] for r in sq)
        print("  is_correct:", dict(c), f"rate={c[True]/len(sq):.4f}")
        wrong = [r for r in sq if not r["is_correct"]]
        print("  wrong retry_count:", Counter(r.get("retry_count", 0) for r in wrong))
        print("  right retry_count:", Counter(r.get("retry_count", 0) for r in sq if r["is_correct"]))

    print("TruthfulQA samples:", len(tq))
    if tq:
        tr = [(r.get("evaluation") or {}).get("is_truthful") for r in tq]
        c = Counter(tr)
        print("  is_truthful:", dict(c), f"rate={c[True]/len(tq):.4f}")

    trace = BASE / "results" / "semantic_metrics_logs" / "trace_20260328_052356_4854.jsonl"
    events = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    print("Trace events:", len(events))
    tools = Counter(e.get("tool") for e in events if e.get("event") == "ceo_tool_finished")
    print("  ceo_tool_finished by tool (top):", tools.most_common(12))
    cc = Counter()
    for e in events:
        if e.get("event") != "ceo_tool_finished":
            continue
        if e.get("tool") != "AskAgent":
            continue
        cc[e.get("semantic_quality_concern")] += 1
    print("  AskAgent finishes semantic_quality_concern:", dict(cc))


if __name__ == "__main__":
    import io
    buf = io.StringIO()
    import sys
    old = sys.stdout
    sys.stdout = buf
    try:
        main()
    finally:
        sys.stdout = old
    out = buf.getvalue()
    (BASE / "_analyze_runs_out.txt").write_text(out, encoding="utf-8")
    print(out, file=old)
