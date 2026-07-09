#!/usr/bin/env python3
"""
AUROC + bootstrap confidence intervals for orchestration semantic metrics.

Predicts whether the **graded final benchmark answer** is incorrect using worker-level
semantic metrics from paper/thesis traces (metrics-on runs only).

Join: trace ``question_index`` ↔ results ``question_num`` (or ``bench_index`` for GSM8K).
MMLU-Pro law: trace ``question_id`` ↔ ``law_result.json`` ``question_id``.

Aggregation (per question, across worker_answer / worker_answer_multi rows):
  - entropy score  = max(semantic_entropy)
  - density score  = -min(semantic_density)   # higher => more likely wrong
  - concern score  = 1.0 if any semantic_quality_concern else 0.0

Outputs (under eval_results/):
  - semantic_auroc_summary.json
  - semantic_auroc_summary.csv
  - semantic_auroc_plot.png
  - semantic_auroc_report.txt

Usage (from HASHIRU_Bench/bench):
  python3 eval_results/compute_semantic_auroc.py
  python3 eval_results/compute_semantic_auroc.py --bootstrap 2000 --min-questions 15
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parents[1]
LOGS = BENCH / "results/semantic_metrics_logs"
EVAL = Path(__file__).resolve().parent

# Paper/thesis metrics-on runs with joinable correctness labels (see _generate_benchmark_ablation_playbyplays.py).
PAPER_RUNS: list[dict[str, Any]] = [
    {
        "key": "gsm8k",
        "label": "GSM8K",
        "trace": LOGS / "trace_20260407_175308_6622.jsonl",
        "results": EVAL / "gsm8k_test_20260407_175331.jsonl",
        "results_kind": "jsonl",
        "join": "question_index_to_num",
        "correct_field": "is_correct",
    },
    {
        "key": "strategyqa",
        "label": "StrategyQA",
        "trace": LOGS / "trace_20260414_094345_2284928.jsonl",
        "results": BENCH / "strategyqa_results/strategyqa_benchmark_20260414_095001.jsonl",
        "results_kind": "jsonl",
        "join": "question_index_to_num",
        "correct_field": "is_correct",
    },
    {
        "key": "truthfulqa",
        "label": "TruthfulQA",
        "trace": LOGS / "trace_20260407_012530_12511.jsonl",
        "results": BENCH / "truthful_qa_results/truthful_qa_benchmark_20260407_012538.jsonl",
        "results_kind": "jsonl",
        "join": "question_index_to_num",
        "correct_field": "_truthfulqa_is_truthful",
    },
    {
        "key": "musique",
        "label": "MuSiQue",
        "trace": LOGS / "trace_20260611_065038_368770.jsonl",
        "results": BENCH / "musique_results/musique_benchmark_20260611_065043.jsonl",
        "results_kind": "jsonl",
        "join": "question_index_to_num",
        "correct_field": "is_correct",
    },
    {
        "key": "hotpotqa",
        "label": "HotpotQA",
        "trace": LOGS / "trace_20260611_092516_552311.jsonl",
        "results": BENCH / "hotpotqa_results/hotpotqa_benchmark_20260611_093915.jsonl",
        "results_kind": "jsonl",
        "join": "question_index_to_num",
        "correct_field": "is_correct",
    },
    {
        "key": "gpqa",
        "label": "GPQA (diamond)",
        "trace": LOGS / "trace_20260611_124314_1055898.jsonl",
        "results": BENCH / "gpqa_results/gpqa_gpqa_diamond_benchmark_20260611_131116.jsonl",
        "results_kind": "jsonl",
        "join": "question_index_to_num",
        "correct_field": "is_correct",
    },
    {
        "key": "mmlu_law",
        "label": "MMLU-Pro (law subset)",
        "trace": LOGS / "trace_20260409_174559_372695.jsonl",
        "results": EVAL / "law_result.json",
        "results_kind": "law_json",
        "join": "question_id",
        "correct_field": "_law_pred_matches_gold",
    },
]

SCORERS = (
    ("entropy", "Semantic entropy (max)", "max_entropy"),
    ("neg_density", "Neg. min density", "neg_min_density"),
    ("concern", "Quality concern (any)", "any_concern"),
)


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
    if buf.strip() and depth == 0:
        recs.append(json.loads(buf))
    return recs


def load_results_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else []
    return load_jsonl(path)


def load_law_results(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    out: dict[int, dict] = {}
    for row in data:
        qid = row.get("question_id")
        if qid is None:
            continue
        out[int(qid)] = row
    return out


def _new_worker_slot() -> dict:
    return {
        "question_id": None,
        "entropy_vals": [],
        "density_vals": [],
        "concerns": [],
        "n_workers": 0,
    }


def _update_worker_slot(slot: dict, event: dict) -> None:
    qid = event.get("question_id")
    if qid is not None:
        slot["question_id"] = str(qid)
    ent = event.get("semantic_entropy")
    den = event.get("semantic_density")
    if ent is not None:
        try:
            slot["entropy_vals"].append(float(ent))
        except (TypeError, ValueError):
            pass
    if den is not None:
        try:
            slot["density_vals"].append(float(den))
        except (TypeError, ValueError):
            pass
    if event.get("semantic_quality_concern") is not None:
        slot["concerns"].append(bool(event["semantic_quality_concern"]))
    slot["n_workers"] += 1


def parse_trace_workers(path: Path) -> tuple[dict[int, dict], dict[str, dict]]:
    """
    Returns:
      by_index: question_index -> aggregated worker metrics (stable-index benchmarks)
      by_qid: question_id -> aggregated worker metrics (required for MMLU-Pro where
              question_index is reused across different items in one trace)
    """
    by_index: dict[int, dict] = defaultdict(_new_worker_slot)
    by_qid_raw: dict[str, dict] = defaultdict(_new_worker_slot)

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("event") not in ("worker_answer", "worker_answer_multi"):
            continue
        qid = o.get("question_id")
        if qid is not None:
            _update_worker_slot(by_qid_raw[str(qid)], o)
        qi = o.get("question_index")
        if qi is not None:
            _update_worker_slot(by_index[int(qi)], o)

    by_index_out: dict[int, dict] = {}
    for qi, slot in by_index.items():
        by_index_out[qi] = _finalize_worker_slot(slot)

    by_qid_out: dict[str, dict] = {}
    for qid, slot in by_qid_raw.items():
        by_qid_out[qid] = _finalize_worker_slot(slot)

    return by_index_out, by_qid_out


def _finalize_worker_slot(slot: dict) -> dict:
    ent = slot["entropy_vals"]
    den = slot["density_vals"]
    con = slot["concerns"]
    return {
        "question_id": slot.get("question_id"),
        "n_workers": slot["n_workers"],
        "max_entropy": max(ent) if ent else None,
        "min_density": min(den) if den else None,
        "neg_min_density": (-min(den)) if den else None,
        "any_concern": (1.0 if any(con) else 0.0) if con else None,
        "has_metrics": bool(ent or den or con),
    }


def result_question_key(row: dict) -> int | None:
    for k in ("question_num", "bench_index"):
        if row.get(k) is not None:
            return int(row[k])
    return None


def is_correct_from_row(row: dict, field: str) -> bool | None:
    if field == "_truthfulqa_is_truthful":
        ev = row.get("evaluation")
        if isinstance(ev, dict) and "is_truthful" in ev:
            return bool(ev["is_truthful"])
        return None
    if field == "_law_pred_matches_gold":
        if "is_correct" in row:
            return bool(row["is_correct"])
        gold = row.get("answer") or row.get("gold") or row.get("label")
        pred = row.get("pred") or row.get("prediction") or row.get("model_pred")
        if gold is None or pred is None:
            return None
        return str(pred).strip().upper() == str(gold).strip().upper()
    if field not in row:
        return None
    return bool(row[field])


def build_labeled_pairs(
    run: dict[str, Any],
    trace_by_index: dict[int, dict],
    trace_by_qid: dict[str, dict],
    results_rows: list[dict] | dict[int, dict],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    join = run["join"]
    field = run["correct_field"]

    if join == "question_index_to_num":
        by_num = {}
        for row in results_rows:  # type: ignore[union-attr]
            qk = result_question_key(row)
            if qk is not None:
                by_num[qk] = row
        for qi, metrics in sorted(trace_by_index.items()):
            res = by_num.get(qi)
            if not res:
                continue
            ok = is_correct_from_row(res, field)
            if ok is None:
                continue
            pairs.append(_make_pair(run["key"], qi, metrics, ok))
    elif join == "question_id":
        law_map: dict[int, dict] = results_rows  # type: ignore[assignment]
        for qid_str, metrics in sorted(
            trace_by_qid.items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
        ):
            try:
                law_row = law_map.get(int(qid_str))
            except (TypeError, ValueError):
                continue
            if not law_row:
                continue
            ok = is_correct_from_row(law_row, field)
            if ok is None:
                continue
            pairs.append(
                _make_pair(
                    run["key"],
                    int(qid_str) if str(qid_str).isdigit() else 0,
                    metrics,
                    ok,
                    question_id=qid_str,
                )
            )
    return pairs


def _make_pair(
    run_key: str,
    key: int,
    metrics: dict,
    is_correct: bool,
    *,
    question_id: str | None = None,
) -> dict[str, Any]:
    return {
        "run_key": run_key,
        "join_key": key,
        "question_id": question_id or metrics.get("question_id"),
        "is_correct": is_correct,
        "incorrect": (not is_correct),
        "max_entropy": metrics.get("max_entropy"),
        "neg_min_density": metrics.get("neg_min_density"),
        "any_concern": metrics.get("any_concern"),
        "n_workers": metrics.get("n_workers", 0),
        "has_metrics": metrics.get("has_metrics", False),
    }


def auroc_score(y_true: list[int], y_score: list[float]) -> float:
    """Higher y_score should predict y_true==1 (incorrect)."""
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        return _auroc_mann_whitney(y_true, y_score)


def _auroc_mann_whitney(y_true: list[int], y_score: list[float]) -> float:
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return float("nan")
    concordant = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                concordant += 1
            elif p == n:
                ties += 1
    return (concordant + 0.5 * ties) / (len(pos) * len(neg))


def bootstrap_auroc_ci(
    y_true: list[int],
    y_score: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (auroc, ci_low, ci_high)."""
    rng = random.Random(seed)
    n = len(y_true)
    if n < 2 or len(set(y_true)) < 2:
        base = auroc_score(y_true, y_score)
        return base, float("nan"), float("nan")
    base = auroc_score(y_true, y_score)
    if math.isnan(base):
        return base, float("nan"), float("nan")
    samples: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        ys = [y_score[i] for i in idx]
        if len(set(yt)) < 2:
            continue
        v = auroc_score(yt, ys)
        if not math.isnan(v):
            samples.append(v)
    if len(samples) < max(50, n_boot // 10):
        return base, float("nan"), float("nan")
    samples.sort()
    lo = samples[int((alpha / 2) * len(samples))]
    hi = samples[int((1 - alpha / 2) * len(samples)) - 1]
    return base, lo, hi


@dataclass
class AurocResult:
    run_key: str
    run_label: str
    scorer_key: str
    scorer_label: str
    n_total: int
    n_used: int
    n_incorrect: int
    n_correct: int
    auroc: float
    ci_low: float
    ci_high: float
    skipped_reason: str | None = None


def analyze_run(
    run: dict[str, Any],
    *,
    n_boot: int,
    min_questions: int,
    seed: int,
) -> tuple[list[AurocResult], list[dict[str, Any]]]:
    trace_path: Path = run["trace"]
    results_path: Path = run["results"]
    if not trace_path.exists():
        return [], []
    if not results_path.exists():
        return [], []

    trace_by_index, trace_by_qid = parse_trace_workers(trace_path)
    if run["results_kind"] == "jsonl":
        results_rows = load_results_rows(results_path)
    else:
        results_rows = load_law_results(results_path)

    pairs = build_labeled_pairs(run, trace_by_index, trace_by_qid, results_rows)
    out: list[AurocResult] = []

    for scorer_key, scorer_label, pair_field in SCORERS:
        usable = [p for p in pairs if p.get(pair_field) is not None]
        n_incorrect = sum(1 for p in usable if p["incorrect"])
        n_correct = len(usable) - n_incorrect
        if len(usable) < min_questions:
            out.append(
                AurocResult(
                    run_key=run["key"],
                    run_label=run["label"],
                    scorer_key=scorer_key,
                    scorer_label=scorer_label,
                    n_total=len(pairs),
                    n_used=len(usable),
                    n_incorrect=n_incorrect,
                    n_correct=n_correct,
                    auroc=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    skipped_reason=f"n_used={len(usable)} < min_questions={min_questions}",
                )
            )
            continue
        if n_incorrect == 0 or n_correct == 0:
            out.append(
                AurocResult(
                    run_key=run["key"],
                    run_label=run["label"],
                    scorer_key=scorer_key,
                    scorer_label=scorer_label,
                    n_total=len(pairs),
                    n_used=len(usable),
                    n_incorrect=n_incorrect,
                    n_correct=n_correct,
                    auroc=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    skipped_reason="single class only",
                )
            )
            continue
        y_true = [1 if p["incorrect"] else 0 for p in usable]
        y_score = [float(p[pair_field]) for p in usable]
        auroc, lo, hi = bootstrap_auroc_ci(y_true, y_score, n_boot=n_boot, seed=seed)
        out.append(
            AurocResult(
                run_key=run["key"],
                run_label=run["label"],
                scorer_key=scorer_key,
                scorer_label=scorer_label,
                n_total=len(pairs),
                n_used=len(usable),
                n_incorrect=n_incorrect,
                n_correct=n_correct,
                auroc=auroc,
                ci_low=lo,
                ci_high=hi,
            )
        )
    return out, pairs


def save_csv(results: list[AurocResult], path: Path) -> None:
    fields = [
        "run_key",
        "run_label",
        "scorer_key",
        "scorer_label",
        "n_total",
        "n_used",
        "n_incorrect",
        "n_correct",
        "auroc",
        "ci_low",
        "ci_high",
        "skipped_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})


def save_json(results: list[AurocResult], path: Path, meta: dict) -> None:
    payload = {
        "meta": meta,
        "results": [{k: getattr(r, k) for k in r.__dataclass_fields__} for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_results(results: list[AurocResult], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    plot_rows = [r for r in results if r.skipped_reason is None and not math.isnan(r.auroc)]
    if not plot_rows:
        return False

    run_labels = []
    seen = set()
    for r in plot_rows:
        if r.run_label not in seen:
            run_labels.append(r.run_label)
            seen.add(r.run_label)

    scorer_labels = [s[1] for s in SCORERS]
    x = np.arange(len(run_labels))
    width = 0.25
    colors = ["#2563eb", "#059669", "#d97706"]

    fig, ax = plt.subplots(figsize=(max(10, len(run_labels) * 1.4), 6))
    for i, (scorer_key, scorer_label, _) in enumerate(SCORERS):
        means = []
        err_lo = []
        err_hi = []
        for rl in run_labels:
            row = next(
                (r for r in plot_rows if r.run_label == rl and r.scorer_key == scorer_key),
                None,
            )
            if row is None:
                means.append(float("nan"))
                err_lo.append(0)
                err_hi.append(0)
            else:
                means.append(row.auroc)
                err_lo.append(max(0, row.auroc - row.ci_low) if not math.isnan(row.ci_low) else 0)
                err_hi.append(max(0, row.ci_high - row.auroc) if not math.isnan(row.ci_high) else 0)
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=[err_lo, err_hi],
            capsize=4,
            label=scorer_label,
            color=colors[i % len(colors)],
            alpha=0.9,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_ylabel("AUROC (predict incorrect final answer)")
    ax.set_xlabel("Benchmark run (paper/thesis, metrics on)")
    ax.set_title(
        "Semantic metrics vs. final-answer correctness\n"
        "(worker metrics aggregated per question; 95% bootstrap CI)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def plot_results_svg(results: list[AurocResult], path: Path) -> bool:
    """Fallback chart when matplotlib is unavailable."""
    plot_rows = [r for r in results if r.skipped_reason is None and not math.isnan(r.auroc)]
    if not plot_rows:
        return False

    run_labels: list[str] = []
    seen: set[str] = set()
    for r in plot_rows:
        if r.run_label not in seen:
            run_labels.append(r.run_label)
            seen.add(r.run_label)

    width, height = max(900, 140 * len(run_labels) + 220), 520
    margin_l, margin_b, margin_t = 220, 120, 80
    group_w = (width - margin_l - 40) / max(len(run_labels), 1)
    bar_w = group_w / (len(SCORERS) + 1)
    colors = ("#2563eb", "#059669", "#d97706")
    y_scale = height - margin_b - margin_t

    def y_pos(v: float) -> float:
        return margin_t + (1.0 - v) * y_scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16" font-family="sans-serif">'
        "Semantic metrics AUROC (predict incorrect final answer)</text>",
        f'<text x="{width/2}" y="48" text-anchor="middle" font-size="12" fill="#555" font-family="sans-serif">'
        "95% bootstrap CI; worker metrics aggregated per question</text>",
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#999"/>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-20}" y2="{height-margin_b}" stroke="#999"/>',
        f'<line x1="{margin_l}" y1="{y_pos(0.5)}" x2="{width-20}" y2="{y_pos(0.5)}" stroke="#ccc" stroke-dasharray="4"/>',
        f'<text x="{margin_l-8}" y="{y_pos(0.5)+4}" text-anchor="end" font-size="11" fill="#666" font-family="sans-serif">0.5</text>',
    ]

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y_pos(tick)
        parts.append(f'<line x1="{margin_l-4}" y1="{yy}" x2="{margin_l}" y2="{yy}" stroke="#999"/>')
        parts.append(
            f'<text x="{margin_l-8}" y="{yy+4}" text-anchor="end" font-size="11" '
            f'font-family="sans-serif">{tick:.2f}</text>'
        )

    for gi, rl in enumerate(run_labels):
        gx = margin_l + gi * group_w + bar_w
        parts.append(
            f'<text x="{gx + bar_w}" y="{height - margin_b + 28}" text-anchor="middle" '
            f'font-size="11" font-family="sans-serif">{rl}</text>'
        )
        for si, (scorer_key, scorer_label, _) in enumerate(SCORERS):
            row = next(
                (r for r in plot_rows if r.run_label == rl and r.scorer_key == scorer_key),
                None,
            )
            if row is None:
                continue
            x = gx + si * bar_w
            h = max(0.0, row.auroc) * y_scale
            y = height - margin_b - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" '
                f'fill="{colors[si]}" opacity="0.9"/>'
            )
            if not math.isnan(row.ci_low) and not math.isnan(row.ci_high):
                y_lo = y_pos(row.ci_low)
                y_hi = y_pos(row.ci_high)
                cx = x + bar_w * 0.425
                parts.append(
                    f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" '
                    f'stroke="#111" stroke-width="2"/>'
                )
                parts.append(f'<line x1="{cx-4:.1f}" y1="{y_hi:.1f}" x2="{cx+4:.1f}" y2="{y_hi:.1f}" stroke="#111" stroke-width="2"/>')
                parts.append(f'<line x1="{cx-4:.1f}" y1="{y_lo:.1f}" x2="{cx+4:.1f}" y2="{y_lo:.1f}" stroke="#111" stroke-width="2"/>')

    lx, ly = width - 180, margin_t + 10
    for si, (_, scorer_label, _) in enumerate(SCORERS):
        parts.append(f'<rect x="{lx}" y="{ly + si*18}" width="12" height="12" fill="{colors[si]}"/>')
        parts.append(
            f'<text x="{lx+18}" y="{ly + si*18 + 10}" font-size="11" font-family="sans-serif">{scorer_label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return True


def plot_results_any(results: list[AurocResult], path: Path) -> str:
    if plot_results(results, path):
        return "matplotlib"
    if plot_results_svg(results, path.with_suffix(".svg")):
        return "svg"
    return "none"


def write_report(results: list[AurocResult], path: Path, meta: dict) -> None:
    lines = [
        "Semantic metric AUROC report",
        "=" * 60,
        f"Target: predict incorrect graded final answer (label=1 if wrong).",
        f"Aggregation: max entropy, -min density, any quality concern per question.",
        f"Bootstrap samples: {meta.get('bootstrap', '?')}, seed={meta.get('seed', '?')}",
        "",
    ]
    current = None
    for r in results:
        if r.run_label != current:
            current = r.run_label
            lines.append(f"\n## {r.run_label} ({r.run_key})")
        if r.skipped_reason:
            lines.append(
                f"  {r.scorer_label}: SKIPPED ({r.skipped_reason}; "
                f"n_joined={r.n_total}, n_used={r.n_used}, "
                f"incorrect={r.n_incorrect}, correct={r.n_correct})"
            )
        else:
            lines.append(
                f"  {r.scorer_label}: AUROC={r.auroc:.3f} "
                f"[{r.ci_low:.3f}, {r.ci_high:.3f}] "
                f"(n={r.n_used}, wrong={r.n_incorrect}, right={r.n_correct})"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="AUROC for semantic metrics on paper/thesis traces.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap replicates for 95%% CI.")
    parser.add_argument("--min-questions", type=int, default=15, help="Minimum joined questions per run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EVAL,
        help="Output directory (default: eval_results/).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[AurocResult] = []
    skipped_runs: list[str] = []

    for run in PAPER_RUNS:
        if not run["trace"].exists() or not run["results"].exists():
            skipped_runs.append(f"{run['key']}: missing trace or results file")
            continue
        res, _ = analyze_run(
            run,
            n_boot=args.bootstrap,
            min_questions=args.min_questions,
            seed=args.seed,
        )
        all_results.extend(res)

    meta = {
        "bootstrap": args.bootstrap,
        "min_questions": args.min_questions,
        "seed": args.seed,
        "target": "incorrect final benchmark answer",
        "aggregation": "max_entropy, neg_min_density, any_concern",
        "runs_included": [r["key"] for r in PAPER_RUNS],
        "skipped_missing_files": skipped_runs,
    }

    csv_path = args.out_dir / "semantic_auroc_summary.csv"
    json_path = args.out_dir / "semantic_auroc_summary.json"
    report_path = args.out_dir / "semantic_auroc_report.txt"
    plot_path = args.out_dir / "semantic_auroc_plot.png"

    save_csv(all_results, csv_path)
    save_json(all_results, json_path, meta)
    write_report(all_results, report_path, meta)
    plotted = plot_results_any(all_results, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    if plotted == "matplotlib":
        print(f"Wrote {plot_path} and {plot_path.with_suffix('.pdf')}")
    elif plotted == "svg":
        print(f"Wrote {plot_path.with_suffix('.svg')} (matplotlib not installed)")
    else:
        print("Plot skipped (no plottable rows).")

    if skipped_runs:
        print("\nMissing files (runs omitted):")
        for s in skipped_runs:
            print(f"  - {s}")

    print("\nSummary:")
    for r in all_results:
        if r.scorer_key != "entropy":
            continue
        if r.skipped_reason:
            print(f"  {r.run_label}: SKIPPED ({r.skipped_reason})")
        else:
            print(
                f"  {r.run_label}: entropy AUROC={r.auroc:.3f} "
                f"[{r.ci_low:.3f}, {r.ci_high:.3f}] n={r.n_used}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
