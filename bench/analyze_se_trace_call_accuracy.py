#!/usr/bin/env python3
"""
Build accuracy × worker-call-count tables from SE/SD benchmark JSONL + orchestration traces.

Worker calls per question: each ceo_tool_finished AskAgent = 1; AskMultipleAgents =
called_agent_count (or len(per_agent_outputs)).

Matching trace rows to benchmark rows (in order):
  1. ``benchmark_name`` + ``question_index`` on trace lines (same as benchmark ``question_num``
     / row order), when present.
  2. ``question_text`` or ``user_turn_excerpt`` on trace lines vs benchmark question / jailbreak goal.
  3. Worker prompt substring / token-overlap heuristics (legacy).

Usage:
  python HASHIRU_Bench/bench/analyze_se_trace_call_accuracy.py TRACE_FILENAME.jsonl
  python .../analyze_se_trace_call_accuracy.py trace_20260401_075854_1254.jsonl --results strategyqa_results/foo.jsonl
  python ... > results/se_call_accuracy_tables.md
  python ... --all   # bundled three-job report (hardcoded paths)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_pretty_json_array(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    pos = 0
    out: list[dict] = []
    n = len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        obj, end = dec.raw_decode(text, pos)
        if isinstance(obj, dict):
            out.append(obj)
        pos = end
    return out


def load_jsonl_objects(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            rows.append(o)
    return rows


def load_trace_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def load_benchmark_results(path: Path) -> list[dict]:
    if "jailbreak" in path.name.lower():
        return load_jsonl_objects(path)
    return load_pretty_json_array(path)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _is_correct_result(r: dict, correct_key: str) -> bool:
    v = r.get(correct_key)
    if v is not None:
        return bool(v)
    ev = r.get("evaluation")
    if isinstance(ev, dict) and correct_key in ev:
        return bool(ev[correct_key])
    return False


def extract_match_text_from_ceo_tool(e: dict) -> str:
    if e.get("event") != "ceo_tool_finished":
        return ""
    t = e.get("tool")
    args = e.get("args") if isinstance(e.get("args"), dict) else {}
    if t == "AskAgent":
        return str(e.get("worker_prompt") or args.get("prompt") or "")
    if t == "AskMultipleAgents":
        uq = args.get("user_question")
        if isinstance(uq, str) and uq.strip():
            return uq
        raw = args.get("agent_prompts_json")
        if isinstance(raw, str) and raw.strip():
            try:
                arr = json.loads(raw)
                if isinstance(arr, list) and arr:
                    p0 = arr[0].get("prompt") if isinstance(arr[0], dict) else None
                    if isinstance(p0, str):
                        return p0
            except json.JSONDecodeError:
                pass
        return ""
    return ""


def qa_prompt_matches_question(prompt: str, question: str) -> bool:
    p = _norm(prompt)
    q = _norm(question)
    if not p or not q:
        return False
    if q in p:
        return True
    if len(q) >= 25 and p in q:
        return True
    q_short = q[:100]
    return q_short in p


def qa_token_overlap_prompt_question(prompt: str, question: str, min_shared: int = 4) -> bool:
    p = _norm(prompt)
    q = _norm(question)
    if not p or not q:
        return False
    pt = set(re.findall(r"[a-z0-9]{4,}", p))
    qt = set(re.findall(r"[a-z0-9]{4,}", q))
    if not qt:
        return False
    shared = len(pt & qt)
    need = min(min_shared, max(3, len(qt) // 4))
    return shared >= need


def _norm_bench_slug(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_")


# slug -> (display_name, correct_key, mode, trace_bench_slug)
BENCHMARK_SLUG_CONFIG: dict[str, tuple[str, str, str, str]] = {
    "strategyqa": ("StrategyQA + SE/SD", "is_correct", "qa", "strategyqa"),
    "truthful_qa": ("TruthfulQA + SE/SD", "is_truthful", "qa", "truthful_qa"),
    "jailbreakbench": ("JailbreakBench + SE/SD", "is_correct", "jailbreak", "jailbreakbench"),
}


def semantic_metrics_logs_dir(bench: Path) -> Path:
    return bench / "results" / "semantic_metrics_logs"


def resolve_trace_path(bench: Path, trace_arg: str) -> Path:
    raw = trace_arg.strip()
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    name = p.name
    return (semantic_metrics_logs_dir(bench) / name).resolve()


def infer_bench_slug_from_trace(trace_path: Path) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    known = frozenset(BENCHMARK_SLUG_CONFIG.keys())
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        bn = o.get("benchmark_name")
        if bn is None:
            continue
        sl = _norm_bench_slug(str(bn))
        if sl in known:
            counts[sl] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda x: (x[1], x[0]))[0]


def find_latest_benchmark_results(bench: Path, slug: str) -> Path | None:
    slug = _norm_bench_slug(slug)
    subdirs = {
        "strategyqa": bench / "strategyqa_results",
        "truthful_qa": bench / "truthful_qa_results",
        "jailbreakbench": bench / "results",
    }
    d = subdirs.get(slug)
    if not d or not d.is_dir():
        return None
    candidates = [
        p
        for p in d.iterdir()
        if p.is_file() and "benchmark" in p.name.lower() and p.suffix in (".jsonl", ".json")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_results_path(bench: Path, arg: str | Path) -> Path:
    p = Path(arg)
    if p.is_absolute():
        r = p.resolve()
        if not r.is_file():
            raise SystemExit(f"Results file not found: {r}")
        return r
    cand = (bench / p).resolve()
    if cand.is_file():
        return cand
    raise SystemExit(f"Results file not found: {cand} (resolve paths relative to {bench})")


def build_single_job(
    bench: Path,
    trace_path: Path,
    results_arg: str | Path | None,
    benchmark: str | None,
) -> tuple[str, Path, Path, str, str, str]:
    if not trace_path.is_file():
        raise SystemExit(f"Trace not found: {trace_path}")
    slug: str | None
    if benchmark:
        slug = _norm_bench_slug(benchmark)
    else:
        slug = infer_bench_slug_from_trace(trace_path)
    if not slug or slug not in BENCHMARK_SLUG_CONFIG:
        raise SystemExit(
            "Could not infer benchmark from trace lines with benchmark_name; "
            "pass --benchmark strategyqa|truthful_qa|jailbreakbench"
        )
    if results_arg is not None:
        rp = resolve_results_path(bench, results_arg)
    else:
        found = find_latest_benchmark_results(bench, slug)
        if found is None:
            raise SystemExit(
                f"No *benchmark*.jsonl/json under results dirs for {slug}; pass --results PATH"
            )
        rp = found
    name, ck, mode, trace_slug = BENCHMARK_SLUG_CONFIG[slug]
    return name, rp, trace_path, ck, mode, trace_slug


def default_jobs(bench: Path) -> list[tuple[str, Path, Path, str, str, str]]:
    return [
        (
            "TruthfulQA + SE/SD",
            bench / "truthful_qa_results/truthful_qa_benchmark_20260402_075543.jsonl",
            bench / "results/semantic_metrics_logs/trace_20260402_075519_21517.jsonl",
            "is_truthful",
            "qa",
            "truthful_qa",
        ),
        (
            "StrategyQA + SE/SD",
            bench / "strategyqa_results/strategyqa_benchmark_20260401_224314.jsonl",
            bench / "results/semantic_metrics_logs/trace_20260401_223708_20974.jsonl",
            "is_correct",
            "qa",
            "strategyqa",
        ),
        (
            "JailbreakBench + SE/SD",
            bench / "results/jailbreakbench_benchmark_20260401_075911.jsonl",
            bench / "results/semantic_metrics_logs/trace_20260401_075854_1254.jsonl",
            "is_correct",
            "jailbreak",
            "jailbreakbench",
        ),
    ]


def add_trace_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "trace",
        nargs="?",
        help="Trace JSONL filename in results/semantic_metrics_logs/ (basename is enough)",
    )
    parser.add_argument(
        "--results",
        default=None,
        metavar="PATH",
        help="Benchmark results file, path relative to bench/ or absolute (default: newest *benchmark* file for that benchmark)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["strategyqa", "truthful_qa", "jailbreakbench"],
        default=None,
        help="Benchmark when trace lacks benchmark_name or to pick results dir",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run three bundled example jobs (fixed paths); ignores trace",
    )


def jobs_from_args(bench: Path, args: argparse.Namespace) -> list[tuple[str, Path, Path, str, str, str]]:
    if args.all:
        return default_jobs(bench)
    if not args.trace:
        raise SystemExit(
            "Pass the trace filename (lives under results/semantic_metrics_logs/), or use --all.\n"
            "Example: python analyze_se_trace_call_accuracy.py trace_20260401_075854_1254.jsonl"
        )
    tp = resolve_trace_path(bench, args.trace)
    ra: str | Path | None = args.results
    job = build_single_job(bench, tp, ra, args.benchmark)
    return [job]


def _worker_call_increment(e: dict) -> int:
    t = e.get("tool")
    if t == "AskAgent":
        return 1
    if t == "AskMultipleAgents":
        k = e.get("called_agent_count")
        if k is None:
            pa = e.get("per_agent_outputs")
            k = len(pa) if isinstance(pa, list) else 0
        return int(k or 0)
    return 0


def build_worker_events_by_bench_question(
    events: list[dict], *, trace_bench_slug: str
) -> dict[tuple[str, int], list[dict]]:
    want = _norm_bench_slug(trace_bench_slug)
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for e in events:
        if e.get("event") != "ceo_tool_finished":
            continue
        if e.get("tool") not in ("AskAgent", "AskMultipleAgents"):
            continue
        bn = e.get("benchmark_name")
        qi = e.get("question_index")
        if bn is None or qi is None:
            continue
        if _norm_bench_slug(str(bn)) != want:
            continue
        try:
            qi_int = int(qi)
        except (TypeError, ValueError):
            continue
        out[(want, qi_int)].append(e)
    return out


def _jailbreak_excerpt_or_trace_text(e: dict, result: dict) -> bool:
    key = jailbreak_match_text(result)
    ex = _norm(e.get("user_turn_excerpt") or "")
    qt = _norm(str(e.get("question_text") or ""))
    if not key:
        return False
    if key in ex or key in qt:
        return True
    if len(key) > 40 and key[: min(60, len(key))] in ex:
        return True
    return False


def jailbreak_match_text(result: dict) -> str:
    inp = result.get("input") or ""
    if not isinstance(inp, str):
        return ""
    # Goal usually after the CEO mandate block
    parts = inp.split("\n\n")
    tail = parts[-1] if parts else inp
    return _norm(tail[-800:])


def jailbreak_prompt_matches(prompt: str, result: dict) -> bool:
    key = jailbreak_match_text(result)
    p = _norm(prompt)
    if not key or not p:
        return False
    if key in p:
        return True
    if len(key) > 40 and key[: min(60, len(key))] in p:
        return True
    # overlap on alphanumeric runs
    key_toks = set(re.findall(r"[a-z0-9]{5,}", key))
    p_toks = set(re.findall(r"[a-z0-9]{5,}", p))
    return len(key_toks & p_toks) >= 3


def get_matched_worker_events(
    events: list[dict],
    result: dict,
    *,
    mode: str,
    trace_bench_slug: str,
    row_index_1based: int,
) -> tuple[list[dict], bool]:
    """
    Return ``ceo_tool_finished`` rows for AskAgent / AskMultipleAgents aligned to this
    benchmark result (same matching rules as worker call counting).
    """
    slug = _norm_bench_slug(trace_bench_slug)
    qn = result.get("question_num")
    if qn is None:
        qn = row_index_1based
    try:
        qn = int(qn)
    except (TypeError, ValueError):
        qn = row_index_1based

    by_bq = build_worker_events_by_bench_question(events, trace_bench_slug=trace_bench_slug)
    indexed = by_bq.get((slug, qn), [])
    if indexed:
        return list(indexed), True

    picked: list[dict] = []
    q = str(result.get("question") or "")

    for e in events:
        if e.get("event") != "ceo_tool_finished":
            continue
        t = e.get("tool")
        if t not in ("AskAgent", "AskMultipleAgents"):
            continue
        pr = extract_match_text_from_ceo_tool(e)
        ok = False
        if mode == "jailbreak":
            if jailbreak_prompt_matches(pr, result):
                ok = True
            elif _jailbreak_excerpt_or_trace_text(e, result):
                ok = True
        else:
            if qa_prompt_matches_question(pr, q):
                ok = True
            else:
                qt = e.get("question_text")
                if qt and qa_prompt_matches_question(str(qt), q):
                    ok = True
                else:
                    ex = _norm(e.get("user_turn_excerpt") or "")
                    if q and len(_norm(q)) >= 12 and _norm(q) in ex:
                        ok = True
                    elif q and qa_token_overlap_prompt_question(pr, q):
                        ok = True
        if ok:
            picked.append(e)
    return picked, bool(picked)


def count_worker_calls_for_result(
    events: list[dict],
    result: dict,
    *,
    mode: str,
    trace_bench_slug: str,
    row_index_1based: int,
) -> tuple[int, bool]:
    """Returns (call_count, any_matching_tool_seen)."""
    evs, matched = get_matched_worker_events(
        events,
        result,
        mode=mode,
        trace_bench_slug=trace_bench_slug,
        row_index_1based=row_index_1based,
    )
    if not matched:
        return 0, False
    return sum(_worker_call_increment(e) for e in evs), True


def table_md(rows: dict[int, dict[str, int]], title: str) -> str:
    keys = sorted(rows.keys())
    lines = [
        f"### {title}",
        "",
        "| Worker calls | Correct | Incorrect | Total |",
        "|---:|---:|---:|---:|",
    ]
    for k in keys:
        c = rows[k]
        tot = c.get("correct", 0) + c.get("incorrect", 0)
        lines.append(
            f"| {k} | {c.get('correct', 0)} | {c.get('incorrect', 0)} | {tot} |"
        )
    tc = sum(rows[k].get("correct", 0) for k in keys)
    ti = sum(rows[k].get("incorrect", 0) for k in keys)
    lines.append(f"| **All** | **{tc}** | **{ti}** | **{tc + ti}** |")
    return "\n".join(lines)


def run_one(
    name: str,
    results_path: Path,
    trace_path: Path,
    correct_key: str,
    mode: str,
    trace_bench_slug: str,
) -> tuple[str, dict[int, dict[str, int]], int, int]:
    results = load_benchmark_results(results_path)
    events = load_trace_events(trace_path)

    grid: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    no_match = 0
    for i, r in enumerate(results):
        calls, matched = count_worker_calls_for_result(
            events,
            r,
            mode=mode,
            trace_bench_slug=trace_bench_slug,
            row_index_1based=i + 1,
        )
        if not matched:
            no_match += 1
            continue
        bucket = calls if calls >= 1 else 0
        ok = _is_correct_result(r, correct_key)
        if ok:
            grid[bucket]["correct"] += 1
        else:
            grid[bucket]["incorrect"] += 1

    md = table_md(
        dict(grid),
        f"{name} (trace aligned: {len(results) - no_match} of {len(results)} rows)",
    )
    if no_match:
        md += f"\n\n*Rows with **no** matching AskAgent/AskMultipleAgents prompt in trace: **{no_match}**.*\n"
    return md, dict(grid), len(results), no_match


def main() -> None:
    bench = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Accuracy × worker-call-count tables from benchmark JSONL + orchestration trace."
    )
    add_trace_cli_arguments(parser)
    args = parser.parse_args()
    jobs = jobs_from_args(bench, args)

    combined: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    sections: list[str] = []

    for name, rp, tp, ck, mode, slug in jobs:
        if not rp.is_file():
            sections.append(f"### {name}\n\n*Missing results: {rp}*\n")
            continue
        if not tp.is_file():
            sections.append(f"### {name}\n\n*Missing trace: {tp}*\n")
            continue
        md, grid, _, _ = run_one(name, rp, tp, ck, mode, slug)
        sections.append(md)
        for b, c in grid.items():
            combined[b]["correct"] += c.get("correct", 0)
            combined[b]["incorrect"] += c.get("incorrect", 0)

    sections.append(
        table_md(
            dict(combined),
            "Combined (SE/SD runs, aligned rows only)",
        )
    )
    out = "\n\n".join(sections)
    print(out)
    out_path = bench / "results" / "se_call_accuracy_tables.md"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out + "\n", encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    main()
