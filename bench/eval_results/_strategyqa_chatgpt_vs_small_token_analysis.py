#!/usr/bin/env python3
"""
Reproducible StrategyQA ablation comparison: ChatGPT workers vs small open-weight workers.

See stdout / strategyqa_chatgpt_vs_small_token_analysis.md for the human-readable report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _EVAL_DIR.parent
_STRATEGYQA_DIR = _BENCH_ROOT / "strategyqa_results"
_TRACE_DIR = _BENCH_ROOT / "results" / "semantic_metrics_logs"
_OUT_MD = _EVAL_DIR / "strategyqa_chatgpt_vs_small_token_analysis.md"

_RE_BENCH_TS = re.compile(r"strategyqa_benchmark_(\d{8})_(\d{6})\.jsonl$")
_RE_TRACE_TS = re.compile(r"trace_(\d{8})_(\d{6})_\d+\.jsonl$")

# Avoid matching fractional timestamps like 1778121548.8727736 (.429…) as quota errors.
_RE_429 = re.compile(
    r"(?i)(?:RESOURCE_EXHAUSTED|quota.?exhaust|exceeded.?your.?current.?quota|rate.?limits?|HTTP\s*\/?\s*429)"
    r"|(?:(?<![.\d])429(?!\d))"
)

_CHATGPT = "chatgpt-5.4"


def _is_small_worker_model(name: str) -> bool:
    if not name:
        return False
    n = name.lower()
    return n.startswith("deepseek-r1") or n.startswith("llama3.2")


def _is_chatgpt_worker_model(name: str) -> bool:
    return bool(name and name.startswith(_CHATGPT))


def iter_benchmark_records(path: Path) -> Iterable[dict]:
    """Parse pretty-printed benchmark JSONL (multiple lines per object)."""
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        yield obj
        i = end


def parse_ts_from_bench_name(name: str) -> Optional[datetime]:
    m = _RE_BENCH_TS.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def parse_ts_from_trace_name(name: str) -> Optional[datetime]:
    m = _RE_TRACE_TS.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def iter_trace_events(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def percentile_p25_p75(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return float(qs[24]), float(qs[74])  # 25th / 75th inclusive


def summarize_dist(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    p25, p75 = percentile_p25_p75(values)
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p25": p25,
        "p75": p75,
    }


def closest_preceding_trace(
    bench_ts: datetime, traces: List[Tuple[datetime, Path]]
) -> Optional[Path]:
    candidates = [(t, p) for t, p in traces if t <= bench_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def classify_base_models_from_trace(trace_path: Path) -> Tuple[str, Counter, List[str]]:
    """
    Returns (label, counter, notes).
    Label: 'chatgpt' | 'small' | 'unknown' | 'mixed_nonstandard'
    Uses worker_answer / worker_answer_multi / ceo_create_agent base_model fields via full-line scan.
    """
    counts: Counter = Counter()
    for ev in iter_trace_events(trace_path):
        evname = ev.get("event")
        if evname in ("worker_answer", "ceo_ask_agent", "ceo_create_agent"):
            bm = ev.get("base_model")
            if bm:
                counts[str(bm)] += 1
        if evname == "worker_answer_multi":
            # Embedded models in JSON string
            blob = json.dumps(ev, ensure_ascii=False)
            for m in re.findall(r'"base_model"\s*:\s*"([^"]+)"', blob):
                counts[m] += 1
    notes: List[str] = []
    if not counts:
        return "unknown", counts, ["no base_model in trace"]

    n_small_calls = sum(n for bm, n in counts.items() if _is_small_worker_model(bm))
    n_chat_calls = sum(n for bm, n in counts.items() if _is_chatgpt_worker_model(bm))

    if n_chat_calls > 0 and n_small_calls == 0:
        return "chatgpt", counts, notes
    if n_small_calls > 0 and n_chat_calls == 0:
        return "small", counts, notes
    if n_small_calls > 0 and n_chat_calls > 0:
        notes.append(f"mixed small={n_small_calls} chatgpt={n_chat_calls}")
        dominant, _ = counts.most_common(1)[0]
        if _is_chatgpt_worker_model(str(dominant)):
            return "chatgpt", counts, notes
        if _is_small_worker_model(str(dominant)):
            return "small", counts, notes
        return "mixed_nonstandard", counts, notes
    return "unknown", counts, notes


def build_session_delta_series(trace_path: Path) -> Dict[int, Tuple[int, int]]:
    """
    Per question_index: (delta_in, raw_out_at_question).

    - **Input** is per-question delta: prefer ``ceo_turn_delta_input_tokens``; else
      ``max(0, ceo_session_input_tokens_now - ceo_session_input_tokens_prev)`` over
      ``ceo_final_answer`` rows sorted by timestamp.
    - **Output** is reported raw: ``ceo_session_output_tokens`` at that question
      (the latest ``ceo_final_answer`` for that local question's bench_attempt/ts).
      Per request, no differencing on output.

    The output value is therefore a cumulative session counter at the moment that
    question finished; aggregate stats reflect the magnitude reached, not per-turn
    work. See the `Note on output semantics` section in the report.
    """
    finals: List[dict] = []
    for ev in iter_trace_events(trace_path):
        if ev.get("event") != "ceo_final_answer":
            continue
        if ev.get("benchmark_name") != "strategyqa":
            continue
        finals.append(ev)
    finals.sort(key=lambda e: (e.get("ts") or 0.0, e.get("question_index") or 0, e.get("bench_attempt") or 0))

    prev_in = 0
    by_q: Dict[int, Tuple[int, int]] = {}

    for ev in finals:
        qi = ev.get("question_index")
        if qi is None:
            continue
        qi = int(qi)
        cin_ev = ev.get("ceo_session_input_tokens")
        cout_ev = ev.get("ceo_session_output_tokens")
        delta_in_field = ev.get("ceo_turn_delta_input_tokens")

        if delta_in_field is not None:
            di = int(delta_in_field)
        elif cin_ev is not None:
            cin_i = int(cin_ev)
            di = max(0, cin_i - prev_in)
        else:
            continue

        if cout_ev is not None:
            raw_out = int(cout_ev)
        else:
            d_out = ev.get("ceo_turn_delta_output_tokens")
            if d_out is None:
                continue
            raw_out = int(d_out)

        by_q[qi] = (di, raw_out)
        if cin_ev is not None:
            prev_in = int(cin_ev)
    return by_q


def monotonic_sessions_evidence(trace_path: Path) -> Tuple[bool, bool, List[str]]:
    """Check ceo_session_* monotonicity on ceo_final_answer strategyqa events."""
    ins: List[int] = []
    outs: List[int] = []
    seq: List[dict] = []
    for ev in iter_trace_events(trace_path):
        if ev.get("event") != "ceo_final_answer":
            continue
        if ev.get("benchmark_name") != "strategyqa":
            continue
        if ev.get("ceo_session_input_tokens") is None:
            continue
        seq.append(ev)
    seq.sort(key=lambda e: e.get("ts") or 0.0)
    for ev in seq:
        ins.append(int(ev["ceo_session_input_tokens"]))
        outs.append(int(ev["ceo_session_output_tokens"]))
    lines: List[str] = []
    ok_in = all(ins[i] <= ins[i + 1] for i in range(len(ins) - 1)) if len(ins) > 1 else True
    ok_out = all(outs[i] <= outs[i + 1] for i in range(len(outs) - 1)) if len(outs) > 1 else True
    if ins:
        lines.append(
            f"session_input_tokens: n={len(ins)}, monotonic_nondecreasing={ok_in}, "
            f"first={ins[0]}, last={ins[-1]}"
        )
    if outs:
        lines.append(
            f"session_output_tokens: n={len(outs)}, monotonic_nondecreasing={ok_out}, "
            f"first={outs[0]}, last={outs[-1]}"
        )
    return ok_in, ok_out, lines


def events_for_strategyqa_question(trace_path: Path, question_index: int) -> List[dict]:
    out: List[dict] = []
    for ev in iter_trace_events(trace_path):
        if ev.get("benchmark_name") != "strategyqa":
            continue
        if ev.get("question_index") == question_index:
            out.append(ev)
    return out


def event_has_quota_signal(ev: dict) -> bool:
    blob = json.dumps(ev, ensure_ascii=False)
    return bool(_RE_429.search(blob))


def profile_benchmark(path: Path) -> dict:
    recs = list(iter_benchmark_records(path))
    nums = [int(r["question_num"]) for r in recs if r.get("question_num") is not None]
    correct = sum(1 for r in recs if r.get("is_correct"))
    keys_union: Set[str] = set()
    for r in recs[:5]:
        keys_union.update(r.keys())
    tokenish = sorted(k for k in keys_union if "token" in k.lower())
    te = [float(r["time_elapsed"]) for r in recs if r.get("time_elapsed") is not None]
    return {
        "path": path,
        "n_questions": len(recs),
        "q_min": min(nums) if nums else None,
        "q_max": max(nums) if nums else None,
        "accuracy": correct / len(recs) if recs else 0.0,
        "mean_time": statistics.mean(te) if te else float("nan"),
        "sample_token_keys": tokenish,
    }


@dataclass
class RunSelection:
    bench_path: Path
    trace_path: Path
    label: str
    bench_ts: datetime
    classification: str
    worker_model_notes: List[str] = field(default_factory=list)


def discover_runs() -> Tuple[List[RunSelection], List[str]]:
    bench_files = sorted(_STRATEGYQA_DIR.glob("strategyqa_benchmark_*.jsonl"))
    traces = []
    for p in sorted(_TRACE_DIR.glob("trace_*.jsonl")):
        ts = parse_ts_from_trace_name(p.name)
        if ts:
            traces.append((ts, p))
    traces.sort(key=lambda x: x[0])
    warnings: List[str] = []
    selections: List[RunSelection] = []

    for bp in bench_files:
        ts = parse_ts_from_bench_name(bp.name)
        if not ts:
            warnings.append(f"skip benchmark (bad name): {bp.name}")
            continue
        tp = closest_preceding_trace(ts, traces)
        if tp is None:
            warnings.append(f"no trace <= {bp.name}; skipped")
            continue
        clf, ctr, clf_notes = classify_base_models_from_trace(tp)
        if clf == "unknown":
            warnings.append(f"{bp.name}: unknown worker models {dict(ctr)}")
        selections.append(
            RunSelection(
                bench_path=bp,
                trace_path=tp,
                label="",
                bench_ts=ts,
                classification=clf,
                worker_model_notes=list(clf_notes),
            )
        )
    selections.sort(key=lambda s: s.bench_ts, reverse=True)
    return selections, warnings


def trace_question_index_to_question_id(trace_path: Path) -> Dict[int, str]:
    """Last ceo_final_answer per question_index uses highest bench_attempt then ts."""
    buckets: Dict[int, List[dict]] = defaultdict(list)
    for ev in iter_trace_events(trace_path):
        if ev.get("event") != "ceo_final_answer":
            continue
        if ev.get("benchmark_name") != "strategyqa":
            continue
        qi = ev.get("question_index")
        if qi is None:
            continue
        buckets[int(qi)].append(ev)
    out: Dict[int, str] = {}
    for qi, lst in buckets.items():
        lst.sort(key=lambda e: (e.get("bench_attempt") or 0, e.get("ts") or 0.0))
        qid = lst[-1].get("question_id")
        if qid is not None:
            out[qi] = str(qid)
    return out


def canonical_question_ids_for_run(sel: RunSelection, lo: int = 0, hi: int = 299) -> Set[int]:
    """StrategyQA canonical row ids appearing in this (benchmark, paired trace)."""
    m = trace_question_index_to_question_id(sel.trace_path)
    out: Set[int] = set()
    for rec in iter_benchmark_records(sel.bench_path):
        qn = rec.get("question_num")
        if qn is None:
            continue
        qs = m.get(int(qn))
        if qs is None:
            continue
        cid = int(qs)
        if lo <= cid <= hi:
            out.add(cid)
    return out


def greedy_cover_runs(pool: List[RunSelection], k: int = 3, target_end: int = 300) -> Tuple[List[RunSelection], Set[int]]:
    """
    Repeatedly pick a run maximizing new canonical StrategyQA question_ids in [lo, hi]
    (inclusive endpoints; default corresponds to dataset ids 0..299 — the first 300 rows).
    Pool should be newest-first so ties favor more recent runs.
    """
    lo, hi = 0, target_end - 1
    want = set(range(lo, hi + 1))
    picked: List[RunSelection] = []
    covered: Set[int] = set()

    while len(picked) < k:
        best: Optional[RunSelection] = None
        best_gain = -1
        for s in pool:
            if s in picked:
                continue
            qs = canonical_question_ids_for_run(s, lo, hi)
            gain = len(qs - covered)
            if gain > best_gain:
                best_gain = gain
                best = s
        if best is None or best_gain <= 0:
            break
        picked.append(best)
        covered |= canonical_question_ids_for_run(best, lo, hi)
    return picked, covered


def pick_three_chatgpt(selections: List[RunSelection]) -> Tuple[List[RunSelection], Set[int], List[str]]:
    chat = [s for s in selections if s.classification == "chatgpt"]
    chat.sort(key=lambda s: s.bench_ts, reverse=True)
    picks, cov = greedy_cover_runs(chat, k=3, target_end=300)
    notes: List[str] = []
    missing = sorted(x for x in range(0, 300) if x not in cov)
    if missing:
        notes.append(
            f"chatgpt greedy cover still missing {len(missing)} canonical StrategyQA ids among 0..299 "
            f"(first gaps: {missing[:25]}{' …' if len(missing)>25 else ''})"
        )
    else:
        notes.append("chatgpt selection covers canonical ids 0..299 (first 300 StrategyQA rows).")
    return picks, cov, notes


def coverage_union_canonical(runs: List[RunSelection]) -> Set[int]:
    cov: Set[int] = set()
    for r in runs:
        cov |= canonical_question_ids_for_run(r, 0, 299)
    return cov


def ceo_trace_has_turn_deltas(trace_path: Path) -> bool:
    for ev in iter_trace_events(trace_path):
        if ev.get("event") != "ceo_final_answer" or ev.get("benchmark_name") != "strategyqa":
            continue
        if ev.get("ceo_turn_delta_input_tokens") is not None:
            return True
    return False


def ceo_trace_has_any_session_tokens(trace_path: Path) -> bool:
    for ev in iter_trace_events(trace_path):
        if ev.get("event") != "ceo_final_answer" or ev.get("benchmark_name") != "strategyqa":
            continue
        if ev.get("ceo_session_input_tokens") is not None:
            return True
    return False


def pick_three_small(selections: List[RunSelection]) -> Tuple[List[RunSelection], Set[int], List[str]]:
    small = [s for s in selections if s.classification == "small"]
    small.sort(key=lambda s: s.bench_ts, reverse=True)
    notes: List[str] = []
    picks, cov = greedy_cover_runs(small, k=3, target_end=300)
    miss = sorted(x for x in range(0, 300) if x not in cov)
    if miss:
        notes.append(
            f"small-model greedy cover missing {len(miss)} canonical ids in 0..299 "
            f"(first gaps: {miss[:25]}{' …' if len(miss) > 25 else ''})"
        )
    else:
        notes.append("small-model selection covers canonical StrategyQA ids 0..299 (first 300 rows).")
    with_delta_picked = [s for s in picks if ceo_trace_has_turn_deltas(s.trace_path)]
    without_any_tok = [s for s in picks if not ceo_trace_has_any_session_tokens(s.trace_path)]
    notes.append(
        f"Picked traces with `ceo_turn_delta_*`: {[x.trace_path.name for x in with_delta_picked]}; "
        f"picked runs lacking ceo token fields: {[x.trace_path.name for x in without_any_tok]}."
    )
    return picks, cov, notes


def merge_benchmarks_by_canonical_id(runs_sorted_newest_first: List[RunSelection]) -> Dict[int, dict]:
    """
    Canonical StrategyQA benchmark id (`question_id` in traces) -> merged record.
    Older runs overwritten by newer (`runs_sorted_newest_first` newest-first ordering).
    """
    by_id: Dict[int, dict] = {}
    for s in reversed(runs_sorted_newest_first):
        mmap = trace_question_index_to_question_id(s.trace_path)
        for rec in iter_benchmark_records(s.bench_path):
            q = rec.get("question_num")
            if q is None:
                continue
            qi = int(q)
            qs = mmap.get(qi)
            if qs is None:
                continue
            cid = int(qs)
            if not (0 <= cid <= 299):
                continue
            dr = dict(rec)
            dr["_source_bench"] = s.bench_path.name
            dr["_source_trace"] = s.trace_path.name
            dr["_local_question_num"] = qi
            by_id[cid] = dr
    return by_id


def run_accuracy_and_429(
    by_canon: Dict[int, dict], trace_by_bench_name: Dict[str, Path]
) -> Tuple[int, int, int, List[int]]:
    """correct, total (present canonical ids ⊂ [0..299]), excluded_wrong_AND_429, excluded ids."""
    correct = 0
    total = 0
    excluded = 0
    excluded_nums: List[int] = []
    for cid in range(0, 300):
        rec = by_canon.get(cid)
        if not rec:
            continue
        total += 1
        bench_name = rec.get("_source_bench")
        tp = trace_by_bench_name.get(bench_name)
        wrong = not rec.get("is_correct")
        local_q = int(rec["_local_question_num"])
        has429 = False
        if tp is not None:
            for ev in events_for_strategyqa_question(tp, local_q):
                if event_has_quota_signal(ev):
                    has429 = True
                    break
        blob = json.dumps(rec, ensure_ascii=False)
        if not has429 and _RE_429.search(blob):
            has429 = True
        if wrong and has429:
            excluded += 1
            excluded_nums.append(cid)
            continue
        if rec.get("is_correct"):
            correct += 1
    return correct, total, excluded, excluded_nums


def token_stats_for_merged(
    by_canon: Dict[int, dict], runs: List[RunSelection]
) -> Tuple[List[float], List[float], int, int, Dict[str, Any]]:
    """Returns (delta_input_per_q, raw_output_session_at_q, n_with_tokens, skipped, meta)."""
    di_list, raw_out_list = [], []
    skipped = 0
    bench_to_trace = {s.bench_path.name: s.trace_path for s in runs}
    delta_cache: Dict[str, Dict[int, Tuple[int, int]]] = {}

    per_file_fields: Dict[str, str] = {}
    for s in runs:
        tag = []
        saw_delta = saw_sess = False
        for ev in iter_trace_events(s.trace_path):
            if ev.get("event") != "ceo_final_answer" or ev.get("benchmark_name") != "strategyqa":
                continue
            if ev.get("ceo_turn_delta_input_tokens") is not None:
                saw_delta = True
            if ev.get("ceo_session_input_tokens") is not None:
                saw_sess = True
        if saw_delta:
            tag.append("ceo_turn_delta_*")
        if saw_sess:
            tag.append("ceo_session_*")
        if not tag:
            tag.append("(no token fields on ceo_final_answer)")
        per_file_fields[s.trace_path.name] = ", ".join(tag)

        delta_cache[s.trace_path.name] = build_session_delta_series(s.trace_path)

    for cid in range(0, 300):
        rec = by_canon.get(cid)
        if not rec:
            continue
        bn = rec.get("_source_bench")
        tp = bench_to_trace.get(bn)
        if not tp:
            skipped += 1
            continue
        local_q = int(rec["_local_question_num"])
        dmap = delta_cache.get(tp.name, {})
        tup = dmap.get(local_q)
        if tup is None:
            skipped += 1
            continue
        di_list.append(float(tup[0]))
        raw_out_list.append(float(tup[1]))
    meta = {"per_trace_token_fields": per_file_fields}
    return di_list, raw_out_list, len(di_list), skipped, meta


def _profile_range(r: RunSelection) -> str:
    p = profile_benchmark(r.bench_path)
    return f"n={p['n_questions']}, q=[{p['q_min']}..{p['q_max']}], acc={p['accuracy']:.4f}"


def format_md(
    *,
    chat_runs: List[RunSelection],
    small_runs: List[RunSelection],
    chat_cov: Set[int],
    small_cov: Set[int],
    w: List[str],
    inventory_md: str,
) -> str:
    lines = [
        "# StrategyQA: ChatGPT vs small workers (token + accuracy)",
        "",
        "**Dataset merge key.** Each shard repeats local `question_num` 1…100. Rows are stitched using trace `question_id` "
        "from the winning `ceo_final_answer` (`bench_attempt`, then `ts`). Canonical ints **0…299** = first 300 StrategyQA IDs in this sweep.",
        "",
        inventory_md,
        "",
        "## File pairings",
        "",
        "| Ablation | Benchmark JSONL | Trace JSONL | Classification |",
        "|----------|-----------------|-------------|----------------|",
    ]

    def row(ab: str, s: RunSelection):
        return f"| {ab} | `{s.bench_path.name}` | `{s.trace_path.name}` | `{s.classification}` |"

    for s in chat_runs:
        lines.append(row("chatgpt", s))
    for s in small_runs:
        lines.append(row("small", s))

    lines.extend(["", "## Warnings / discovery", "", *[f"- {x}" for x in w]])

    def cov_block(title: str, runs: List[RunSelection], cov: Set[int]) -> None:
        sub = "\n".join(f"- `{r.bench_path.name}`: {_profile_range(r)}" for r in runs)
        mi = min(cov) if cov else None
        ma = max(cov) if cov else None
        missing = sorted(x for x in range(0, 300) if x not in cov)
        gaps = missing[:40]
        suf = " …" if len(missing) > 40 else ""
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                sub,
                "",
                f"- Union covers StrategyQA canonical ids (trace `question_id`) min={mi} max={ma}, count={len(cov)} "
                f"(target 0..299 = first 300 rows; human ordinal +1)",
                f"- Missing canonical ids in 0..299 (first 40 shown): {gaps}{suf}",
                "",
            ]
        )

    lines.append("## Question coverage")
    cov_block("ChatGPT ablation", chat_runs, chat_cov)
    cov_block("Small-model ablation", small_runs, small_cov)
    return "\n".join(lines)


def build_inventory_md() -> str:
    rows_b = []
    for p in sorted(_STRATEGYQA_DIR.glob("strategyqa_benchmark_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        rows_b.append(
            f"| `{p.name}` | {st.st_size} | {datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')} |"
        )
    rows_t = []
    for p in sorted(_TRACE_DIR.glob("trace_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        rows_t.append(
            f"| `{p.name}` | {st.st_size} | {datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')} |"
        )
    return "\n".join(
        [
            "## Discovery (all files, mtime desc)",
            "",
            "### StrategyQA benchmark JSONLs",
            "",
            "| File | Size (bytes) | mtime (local) |",
            "|------|-------------|---------------|",
            *rows_b,
            "",
            "### Trace JSONLs",
            "",
            "| File | Size (bytes) | mtime (local) |",
            "|------|-------------|---------------|",
            *rows_t,
        ]
    )


def accuracy_per_run_canonical_slice(s: RunSelection) -> Tuple[int, int, int]:
    """Per single benchmark file + paired trace — accuracy restricted to canonical ids 0..299."""
    mq = merge_benchmarks_by_canonical_id([s])
    tb = {s.bench_path.name: s.trace_path}
    cor, tot, exc, _ = run_accuracy_and_429(mq, tb)
    return cor, tot, exc


def full_report_sections(
    chat_runs: List[RunSelection],
    small_runs: List[RunSelection],
    warnings: List[str],
    inventory_md: str,
) -> str:
    sections: List[str] = []
    cov_c = coverage_union_canonical(chat_runs)
    cov_s = coverage_union_canonical(small_runs)

    sections.append(
        format_md(
            chat_runs=chat_runs,
            small_runs=small_runs,
            chat_cov=cov_c,
            small_cov=cov_s,
            w=warnings,
            inventory_md=inventory_md,
        )
    )

    # Accuracy tables
    chat_sorted = sorted(chat_runs, key=lambda x: x.bench_ts, reverse=True)
    small_sorted = sorted(small_runs, key=lambda x: x.bench_ts, reverse=True)

    def per_run_rows(runs: List[RunSelection], name: str) -> List[str]:
        rows = []
        for s in runs:
            recs = list(iter_benchmark_records(s.bench_path))
            c = sum(1 for r in recs if r.get("is_correct"))
            rows.append(f"| `{s.bench_path.name}` | {len(recs)} | {c} | {100.0 * c / len(recs):.2f}% |")
        return rows

    # Merged accuracy on canonical StrategyQA ids 0..299
    mq_c = merge_benchmarks_by_canonical_id(chat_sorted)
    mq_s = merge_benchmarks_by_canonical_id(small_sorted)
    tb_c = {s.bench_path.name: s.trace_path for s in chat_runs}
    tb_s = {s.bench_path.name: s.trace_path for s in small_runs}

    c_cor, c_tot, c_exc, _ = run_accuracy_and_429(mq_c, tb_c)
    s_cor, s_tot, s_exc, _ = run_accuracy_and_429(mq_s, tb_s)

    sections.append("")
    sections.append("## Accuracy (merged, StrategyQA canonical ids 0..299)")
    sections.append("")
    sections.append("| Ablation | Correct | Total | Overall % | Excluded (wrong+429) | Adjusted denom | Adjusted % |")
    sections.append("|----------|---------|-------|-----------|----------------------|----------------|------------|")

    def acc_row(tag: str, cor: int, tot: int, exc: int):
        adj_den = tot - exc
        adj_pct = 100.0 * cor / adj_den if adj_den else float("nan")
        return f"| {tag} | {cor} | {tot} | {100.0 * cor / tot if tot else 0:.2f}% | {exc} | {adj_den} | {adj_pct:.2f}% |"

    sections.append(acc_row("chatgpt", c_cor, c_tot, c_exc))
    sections.append(acc_row("small", s_cor, s_tot, s_exc))

    sections.append("")
    sections.append("## Per-run accuracy (full file)")
    sections.append("")
    sections.append("### ChatGPT runs")
    sections.append("| File | N | Correct | Acc |")
    sections.append("|------|---|---------|-----|")
    sections.extend(per_run_rows(chat_sorted, "chatgpt"))

    sections.append("")
    sections.append("### Per-run — canonical ids 0..299 only (overall vs 429-filtered)")
    sections.append("| File | Correct | Total | Overall % | Excl (wrong∧429) | Adj denom | Adj % |")
    sections.append("|------|---------|-------|-----------|------------------|-----------|-------|")

    def per_run_q300_rows(runs: List[RunSelection]) -> List[str]:
        rows: List[str] = []
        for s in runs:
            cor, tot, exc = accuracy_per_run_canonical_slice(s)
            adj_den = tot - exc
            adj_pct = 100.0 * cor / adj_den if adj_den else float("nan")
            ov = 100.0 * cor / tot if tot else 0.0
            rows.append(
                f"| `{s.bench_path.name}` | {cor} | {tot} | {ov:.2f}% | {exc} | {adj_den} | {adj_pct:.2f}% |"
            )
        return rows

    sections.extend(per_run_q300_rows(chat_sorted))

    sections.append("")
    sections.append("### Small-model runs")
    sections.append("| File | N | Correct | Acc |")
    sections.append("|------|---|---------|-----|")
    sections.extend(per_run_rows(small_sorted, "small"))
    sections.append("")
    sections.append("### Small-model runs — canonical ids 0..299 only")
    sections.append("| File | Correct | Total | Overall % | Excl (wrong∧429) | Adj denom | Adj % |")
    sections.append("|------|---------|-------|-----------|------------------|-----------|-------|")
    sections.extend(per_run_q300_rows(small_sorted))

    sections.append(
        "## CEO tokens per StrategyQA canonical question (merged 0..299)\n\n"
        "- **Δ input:** per-question increase in CEO session input (Gemini `count_tokens` cumulative), "
        "from `ceo_turn_delta_input_tokens` or differencing successive `ceo_session_input_tokens`.\n"
        "- **Output (raw):** `ceo_session_output_tokens` at each question’s final `ceo_final_answer` "
        "(cumulative GeminiManager heuristic **through that question**), **not** differenced—see caveats."
    )
    sections.append("")

    def tok_block(title: str, di, raw_out, nw_, sk_, meta) -> List[str]:
        sc = summarize_dist(di)
        sco = summarize_dist(raw_out)
        return [
            f"### {title}",
            "",
            f"- Rows with token data: **{nw_}**; skipped (missing CEO token data): **{sk_}** (target slice: 300 canonical ids)",
            "",
            "**Per trace token-field presence (ceo_final_answer / strategyqa)**",
            "",
            *[f"- `{k}`: {v}" for k, v in sorted(meta.get("per_trace_token_fields", {}).items())],
            "",
            "| Metric | mean | median | p25 | p75 |",
            "|--------|------|--------|-----|-----|",
            f"| Δ input (per question) | {sc['mean']:.2f} | {sc['median']:.2f} | {sc['p25']:.2f} | {sc['p75']:.2f} |",
            f"| Output (session cumulative at question end) | {sco['mean']:.2f} | {sco['median']:.2f} | {sco['p25']:.2f} | {sco['p75']:.2f} |",
            "",
        ]

    # Tokens
    dci, dco, nw, sk, meta_c = token_stats_for_merged(mq_c, chat_sorted)
    dsi, dso, nw2, sk2, meta_s = token_stats_for_merged(mq_s, small_sorted)

    sections.extend(tok_block("ChatGPT", dci, dco, nw, sk, meta_c))
    sections.extend(tok_block("Small open-weight", dsi, dso, nw2, sk2, meta_s))

    # Cumulative vs delta empirical
    sections.append("## Cumulative `ceo_session_*` vs per-question deltas")
    sections.append("")
    sections.append(
        "In `GeminiManager`, `self.input_tokens` and `self.output_tokens` start at 0 and only increase via `+=` "
        "(see `HASHIRU_modified/src/manager/manager.py`), so traced `ceo_session_*` snapshots are cumulative "
        "process-lifetime totals, not per-question."
    )
    sections.append("")
    for label, runs in [("chatgpt", chat_sorted), ("small", small_sorted)]:
        sections.append(f"### Empirical check — {label}")
        for s in runs:
            ok_in, ok_out, ev_lines = monotonic_sessions_evidence(s.trace_path)
            sections.append(f"- `{s.trace_path.name}`: input_monotonic={ok_in}, output_monotonic={ok_out}")
            for L in ev_lines:
                sections.append(f"  - {L}")
        sections.append("")

    sections.append("## Caveats")
    sections.append("")
    sections.append(
        f"- **429 / quota regex:** `{_RE_429.pattern}` applied to `json.dumps(event)` per trace event "
        "with `benchmark_name=strategyqa` and matching `question_index`, plus benchmark record JSON."
    )
    sections.append(
        "- **Dedup:** Merged keyed by trace `question_id` (= StrategyQA canonical id). Newer benchmark runs overwrite older ones for the same id."
    )
    sections.append(
        "- **Token skips:** Questions outside a run’s file, or `ceo_final_answer` rows without both delta and "
        "computable session diffs, are omitted from token statistics."
    )
    return "\n".join(sections)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chatgpt-files", nargs="*", default=None, help="Override benchmark basenames")
    ap.add_argument("--small-files", nargs="*", default=None)
    args = ap.parse_args()

    selections, warnings = discover_runs()
    inv = build_inventory_md()

    if args.chatgpt_files:
        name_set = set(args.chatgpt_files)
        chat_runs = [s for s in selections if s.bench_path.name in name_set]
    else:
        chat_runs, _, cn = pick_three_chatgpt(selections)
        warnings.extend(cn)

    if args.small_files:
        name_set = set(args.small_files)
        small_runs = [s for s in selections if s.bench_path.name in name_set]
    else:
        small_runs, _, sn = pick_three_small(selections)
        warnings.extend(sn)

    text = full_report_sections(chat_runs, small_runs, warnings, inv)
    print(text)
    _OUT_MD.write_text(text, encoding="utf-8")
    print(f"\nWrote: {_OUT_MD}", file=os.sys.stderr)


if __name__ == "__main__":
    main()
