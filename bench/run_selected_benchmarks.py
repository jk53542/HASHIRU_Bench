"""
Run selected HASHIRU benchmarks back-to-back from one command.

Designed to be launched from:
  ~/hashiru_modified/HASHIRU_Bench/bench

Default sequence:
  1) JailbreakBench
  2) StrategyQA
  3) TruthfulQA
  4) tau2 (telecom)
  5) tau2 (airline)

Each script prepends ``benchmark_ceo_mandate`` instructions so HASHIRU enforces AskAgent/
AskMultipleAgents (same markers as StrategyQA).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class StepResult:
    name: str
    command: list[str]
    returncode: int
    elapsed_s: float


def _infer_jailbreak_num_samples(max_cap: int = 500, fallback: int = 100) -> int:
    """
    Infer total Jailbreak task count from the same dataset used by
    benchmarking_jailbreakbench.py, then cap at max_cap.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        total = 0
        for split in ("harmful", "benign"):
            if split in ds:
                total += len(ds[split])
        if total <= 0:
            raise ValueError("No harmful/benign samples found in dataset.")
        selected = min(total, max_cap)
        print(
            f"[runner] Jailbreak tasks detected: total={total}. "
            f"Using num_samples={selected} (cap={max_cap})."
        )
        return selected
    except Exception as e:
        print(
            f"[runner] Could not auto-detect Jailbreak size ({e}). "
            f"Falling back to num_samples={fallback}."
        )
        return fallback


def _run_step(name: str, command: list[str], dry_run: bool = False) -> StepResult:
    printable = " ".join(shlex.quote(p) for p in command)
    print(f"\n=== RUNNING: {name} ===")
    print(f"Command: {printable}")

    if dry_run:
        print("[dry-run] Skipping execution.")
        return StepResult(name=name, command=command, returncode=0, elapsed_s=0.0)

    t0 = time.time()
    completed = subprocess.run(command, cwd=os.getcwd())
    elapsed = time.time() - t0
    print(f"=== FINISHED: {name} | returncode={completed.returncode} | elapsed={elapsed:.1f}s ===")
    return StepResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        elapsed_s=elapsed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Jailbreak + StrategyQA + TruthfulQA + tau2(telecom/airline) in one go.",
    )
    parser.add_argument(
        "--jailbreak-num-samples",
        type=int,
        default=None,
        help=(
            "Passed to benchmarking_jailbreakbench.py --num-samples. "
            "Default is auto-detected total Jailbreak tasks, capped at 500."
        ),
    )
    parser.add_argument(
        "--jailbreak-offset",
        type=int,
        default=0,
        help="Passed to benchmarking_jailbreakbench.py --offset (default: 0).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any benchmark exits non-zero.",
    )
    parser.add_argument(
        "--truthful-num-questions",
        type=int,
        default=50,
        help=(
            "Passed to benchmark_truthfullQA.py --num-questions "
            "(default: 50, deterministic contiguous slice)."
        ),
    )
    parser.add_argument(
        "--truthful-offset",
        type=int,
        default=0,
        help="Passed to benchmark_truthfullQA.py --offset (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    py = sys.executable
    jailbreak_samples = (
        args.jailbreak_num_samples
        if args.jailbreak_num_samples is not None
        else _infer_jailbreak_num_samples(max_cap=500, fallback=100)
    )

    steps: list[tuple[str, list[str]]] = [
        (
            "JailbreakBench",
            [
                py,
                "benchmarking_jailbreakbench.py",
                "--num-samples",
                str(jailbreak_samples),
                "--offset",
                str(args.jailbreak_offset),
            ],
        ),
        ("StrategyQA", [py, "benchmarking_strategyQA.py"]),
        (
            "TruthfulQA",
            [
                py,
                "benchmark_truthfullQA.py",
                "--num-questions",
                str(args.truthful_num_questions),
                "--offset",
                str(args.truthful_offset),
            ],
        ),
        ("tau2 Telecom", [py, "benchmarking_tau2.py", "--domain", "telecom"]),
        ("tau2 Airline", [py, "benchmarking_tau2.py", "--domain", "airline"]),
    ]

    results: list[StepResult] = []
    for name, command in steps:
        result = _run_step(name=name, command=command, dry_run=args.dry_run)
        results.append(result)
        if result.returncode != 0 and args.stop_on_error:
            print(f"\nStopping early due to failure in: {name}")
            break

    print("\n===== BENCHMARK BATCH SUMMARY =====")
    failed = 0
    for r in results:
        status = "OK" if r.returncode == 0 else "FAIL"
        if r.returncode != 0:
            failed += 1
        print(f"- {r.name}: {status} (code={r.returncode}, {r.elapsed_s:.1f}s)")

    if failed:
        print(f"\nCompleted with failures: {failed}/{len(results)} step(s) failed.")
        return 1

    print("\nAll selected benchmark steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
