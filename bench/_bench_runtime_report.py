#!/usr/bin/env python3
"""Aggregate latest benchmark runtimes + extrapolate to full dataset sizes."""
from __future__ import annotations

import json
from pathlib import Path

# Repo root = parent of HASHIRU_Bench/
REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "HASHIRU_Bench"
BENCH_SUB = BENCH / "bench"
HASHIRU_RES = BENCH / "HASHIRU_results"


def latest_in_dir(d: Path, pattern: str) -> Path | None:
    if not d.is_dir():
        return None
    hits = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def sum_jsonl_times(path: Path) -> tuple[float, int]:
    total = 0.0
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        for k in ("time_elapsed", "response_time", "elapsed", "duration_sec", "wall_time"):
            if k in o and isinstance(o[k], (int, float)):
                total += float(o[k])
                n += 1
                break
    return total, n


def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def mmlu_law_times(path: Path) -> tuple[float, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return 0.0, 0, 0
    tot = 0.0
    n = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        t = row.get("time_elapsed") or row.get("response_time") or row.get("elapsed")
        if isinstance(t, (int, float)):
            tot += float(t)
            n += 1
    return tot, n, len(data)


def main() -> None:
    out_path = BENCH_SUB / "_bench_runtime_report_out.json"
    rows: list[dict] = []

    # --- StrategyQA (latest single jsonl in bench results) ---
    p = latest_in_dir(BENCH_SUB / "strategyqa_results", "strategyqa_benchmark_*.jsonl")
    if p:
        t, n = sum_jsonl_times(p)
        full = 2290  # ChilleD/StrategyQA test split (standard)
        rows.append(
            {
                "benchmark": "StrategyQA",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": full,
                "extrapolated_full_s": round(t * full / n, 2) if n else None,
                "extrapolated_full_h": round(t * full / n / 3600, 2) if n else None,
            }
        )

    p = latest_in_dir(BENCH_SUB / "truthful_qa_results", "truthful_qa_benchmark_*.jsonl")
    if p:
        t, n = sum_jsonl_times(p)
        full = 817
        rows.append(
            {
                "benchmark": "TruthfulQA",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": full,
                "extrapolated_full_s": round(t * full / n, 2) if n else None,
                "extrapolated_full_h": round(t * full / n / 3600, 2) if n else None,
            }
        )

    p = latest_in_dir(BENCH_SUB / "eval_results", "gsm8k_*.jsonl")
    if p:
        t, n = sum_jsonl_times(p)
        full = 1319
        rows.append(
            {
                "benchmark": "GSM8K",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": full,
                "extrapolated_full_s": round(t * full / n, 2) if n else None,
                "extrapolated_full_h": round(t * full / n / 3600, 2) if n else None,
            }
        )

    # Jailbreak: nested under HASHIRU_results
    jb_cands: list[Path] = []
    jb_root = HASHIRU_RES / "jailbreakbench"
    if jb_root.is_dir():
        for sub in ("hashiru", "base"):
            d = jb_root / sub
            if d.is_dir():
                jb_cands.extend(d.glob("jailbreakbench_benchmark_*.jsonl"))
    p = max(jb_cands, key=lambda x: x.stat().st_mtime) if jb_cands else None
    if p:
        t, n = sum_jsonl_times(p)
        full = 200
        rows.append(
            {
                "benchmark": "JailbreakBench",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": full,
                "extrapolated_full_s": round(t * full / n, 2) if n else None,
                "extrapolated_full_h": round(t * full / n / 3600, 2) if n else None,
            }
        )

    # IFBench: latest hashiru-responses under IFBench/eval
    ifb_root = BENCH_SUB / "IFBench" / "eval"
    p = None
    if ifb_root.is_dir():
        cands = list(ifb_root.rglob("hashiru-responses.jsonl"))
        if cands:
            p = max(cands, key=lambda x: x.stat().st_mtime)
    ifb_test = BENCH_SUB / "IFBench" / "data" / "IFBench_test.jsonl"
    full_ifb = count_lines(ifb_test) if ifb_test.exists() else None
    if p and full_ifb:
        t, n = sum_jsonl_times(p)
        rows.append(
            {
                "benchmark": "IFBench",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": full_ifb,
                "extrapolated_full_s": round(t * full_ifb / n, 2) if n else None,
                "extrapolated_full_h": round(t * full_ifb / n / 3600, 2) if n else None,
            }
        )

    # ToolBench under HASHIRU_Bench/results or HASHIRU_results
    tb_cands: list[Path] = []
    for base in (BENCH / "results" / "toolbench", HASHIRU_RES / "toolbench"):
        if base.is_dir():
            tb_cands.extend(base.glob("**/toolbench_hashiru.jsonl"))
            tb_cands.extend(base.glob("**/result/*.jsonl"))
    p = max(tb_cands, key=lambda x: x.stat().st_mtime) if tb_cands else None
    if p:
        t, n = sum_jsonl_times(p)
        rows.append(
            {
                "benchmark": "ToolBench",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": None,
                "extrapolated_full_s": None,
                "extrapolated_full_h": None,
                "note": "Full corpus size not fixed (HF split / local JSON); linear extrapolation needs chosen reference N.",
            }
        )

    # Tau2
    tau_cands: list[Path] = []
    for base in (
        BENCH_SUB,
        HASHIRU_RES / "tau2",
        HASHIRU_RES / "tau2-bench",
    ):
        if base.is_dir():
            tau_cands.extend(base.glob("**/*tau2*.jsonl"))
    p = max(tau_cands, key=lambda x: x.stat().st_mtime) if tau_cands else None
    if p:
        t, n = sum_jsonl_times(p)
        rows.append(
            {
                "benchmark": "Tau2",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": None,
                "extrapolated_full_s": None,
                "extrapolated_full_h": None,
                "note": "Full runtime = chosen domain task count × trials; define N from tau2 tasks.json + --num_trials.",
            }
        )

    # BFCL meta
    bfcl_cands: list[Path] = []
    rdir = BENCH_SUB / "results" / "bfcl"
    if rdir.is_dir():
        bfcl_cands.extend(rdir.glob("**/meta.json"))
    # Avoid scanning all of HASHIRU_results (can be huge on network drives).
    for sub in ("bfcl", "BFCL"):
        d = HASHIRU_RES / sub
        if d.is_dir():
            bfcl_cands.extend(d.glob("**/meta.json"))
    p = max(bfcl_cands, key=lambda x: x.stat().st_mtime) if bfcl_cands else None
    if p:
        meta = json.loads(p.read_text(encoding="utf-8"))
        wall = float(meta.get("total_wall_seconds") or meta.get("wall_seconds") or 0)
        n = int(meta.get("num_samples") or meta.get("n_completed") or meta.get("n_tasks") or 0)
        rows.append(
            {
                "benchmark": "BFCL",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(wall, 2) if wall else None,
                "subset_n": n,
                "full_n": None,
                "extrapolated_full_s": None,
                "extrapolated_full_h": None,
                "note": "BFCL is category-scoped; full leaderboard N depends on selected categories / collection.",
            }
        )

    p = latest_in_dir(BENCH_SUB / "eval_results", "law_result.json")
    if p and p.exists():
        t, n, tot = mmlu_law_times(p)
        rows.append(
            {
                "benchmark": "MMLU-Pro law",
                "artifact": str(p.relative_to(REPO)),
                "subset_sum_s": round(t, 2),
                "subset_n": n,
                "full_n": tot,
                "extrapolated_full_s": round(t * tot / n, 2) if n and tot and n != tot else (round(t, 2) if t else None),
                "extrapolated_full_h": round(t * tot / n / 3600, 2) if n and tot and n != tot else (round(t / 3600, 2) if t else None),
                "note": None if n != tot else "If all law questions present, extrapolation equals subset total.",
            }
        )

    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
