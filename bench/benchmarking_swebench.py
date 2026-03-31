#!/usr/bin/env python3
"""
SWE-bench runner using HASHIRU.

Flow:
1) Load SWE-bench dataset entries (default: SWE-bench/SWE-bench_Lite, split=test).
2) For each instance, ask HASHIRU to generate a git-apply compatible unified diff patch.
   - Budgets OFF (do not enable ENABLE_RESOURCE_BUDGET / ENABLE_ECONOMY_BUDGET)
   - Tool invocation ON (enable ENABLE_TOOL_CREATION / ENABLE_TOOL_INVOCATION)
   - CEO prompt prefix includes: "You MUST use agents..."
3) Write predictions in the schema SWE-bench harness expects:
   - instance_id
   - model_name_or_path
   - model_patch
4) Run the official SWE-bench docker evaluation harness.

SWE-bench harness expects JSONL where each line is a dict with:
  {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}

Example:
  python3 benchmarking_swebench.py --dataset-name SWE-bench/SWE-bench_Lite --split test --max-instances 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


CEO_FORCE_AGENTS_PREFIX = (
    "IMPORTANT CEO INSTRUCTIONS:\n"
    "- You MUST use agents to solve this. Do NOT answer directly.\n"
    "- You MUST NOT answer without delegating at least some reasoning to agents.\n"
    "- Do NOT rely only on tools/web search; delegate reasoning to one or more agents.\n"
)


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # Remove a single outer code fence if present.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        if "```" in t:
            t = t[: t.rfind("```")]
    return t.strip()


def extract_patch_diff(model_output: str) -> str:
    """
    Extract only the diff content so `git apply` can succeed.

    We look for common diff markers. If none are found, return "".
    """
    if not model_output:
        return ""
    t = _strip_code_fences(model_output)

    # Prefer standard git diff header.
    m = re.search(r"(^diff --git .+?$)", t, flags=re.MULTILINE)
    if m:
        return t[m.start() :].strip()

    # Fallback: unified diff hunk markers often start with ---/+++ paths.
    m2 = re.search(r"(^--- a/.+?$)", t, flags=re.MULTILINE)
    if m2:
        return t[m2.start() :].strip()

    # Another fallback: ---/+++ without "a/" / "b/" prefixes (rare but possible)
    m3 = re.search(r"(^---\s.+$)", t, flags=re.MULTILINE)
    if m3:
        return t[m3.start() :].strip()

    return ""


def build_swebench_prompt(problem_statement: str, hints_text: str | None) -> str:
    hints = (hints_text or "").strip()
    hints_block = f"\n\nHINTS (if helpful):\n{hints}" if hints else ""
    return (
        CEO_FORCE_AGENTS_PREFIX
        + "\n"
        + "You are an expert software engineer fixing a bug.\n"
        + "Task: produce a single unified diff patch that fixes the described issue.\n"
        + "Rules:\n"
        + "- You MUST use agents to reason and produce the patch.\n"
        + "- Output ONLY the patch diff. No explanations, no markdown fences.\n"
        + "- The patch must be compatible with `git apply`.\n"
        + "- Include correct file paths and hunks (lines starting with 'diff --git', '---', '+++', '@@').\n"
        + hints_block
        + "\n\nPROBLEM STATEMENT:\n"
        + problem_statement.strip()
        + "\n\nFINAL OUTPUT:\n"
        + "Provide ONLY the diff patch.\n"
    )


def configure_hashiru(client: Any) -> None:
    """
    Set HASHIRU modes:
      - budgets OFF
      - tool invocation ON
    """
    # Use mode names to avoid compatibility issues with modeIndexes numbering.
    mode_indexes = [
        "ENABLE_AGENT_CREATION",
        "ENABLE_LOCAL_AGENTS",
        "ENABLE_CLOUD_AGENTS",
        "ENABLE_TOOL_CREATION",
        "ENABLE_TOOL_INVOCATION",
        # budgets intentionally omitted
    ]
    client.predict(modeIndexes=mode_indexes, api_name="/update_model")


def load_swebench_dataset(dataset_name: str, split: str, instance_ids: Optional[list[str]]) -> list[dict[str, Any]]:
    """
    Load SWE-bench dataset entries using `datasets`.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Missing dependency: datasets. Install with: pip install datasets") from e

    # `load_dataset` supports both HF datasets and local file paths depending on SWE-bench setup.
    # We keep it simple here and support common HF dataset names.
    ds = load_dataset(dataset_name, split=split)
    rows = list(ds)

    if instance_ids:
        idset = set(instance_ids)
        rows = [r for r in rows if r.get("instance_id") in idset]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate + evaluate SWE-bench patches with HASHIRU.")
    parser.add_argument("--dataset-name", type=str, default="SWE-bench/SWE-bench_Lite", help="HF dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--instance-ids", type=str, nargs="*", default=None, help="Optional list of instance IDs")
    parser.add_argument("--max-instances", type=int, default=10, help="Limit number of instances for this run")
    parser.add_argument("--gradio-url", type=str, default=os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--model-name-or-path", type=str, default="hashiru-swebench", help="Used in predictions JSON")
    parser.add_argument("--predictions-out", type=str, default=None, help="Optional output file (.jsonl)")
    parser.add_argument("--run-id", type=str, default=None, help="SWE-bench run_id (required by harness)")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-instance test timeout (seconds)")
    parser.add_argument("--max-workers", type=int, default=4, help="Harness max_workers")
    parser.add_argument("--swebench-python", type=str, default=sys.executable, help="Python executable to run harness")
    parser.add_argument("--write-only", action="store_true", help="Only write predictions; do not run harness")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset.
    rows = load_swebench_dataset(args.dataset_name, args.split, args.instance_ids)
    if not rows:
        raise SystemExit("No SWE-bench instances loaded. Check dataset-name/split/instance-ids.")
    if args.max_instances and args.max_instances > 0:
        rows = rows[: args.max_instances]

    # Output predictions path.
    if args.predictions_out:
        pred_path = Path(args.predictions_out).expanduser().resolve()
    else:
        out_dir = Path("results") / "swebench"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = (out_dir / f"swebench_predictions_{run_id}.jsonl").resolve()

    # Prepare HASHIRU client.
    try:
        from gradio_client import Client
    except ImportError as e:
        raise SystemExit("Missing dependency: gradio_client. Install with: pip install gradio_client") from e

    client = Client(args.gradio_url.rstrip("/") + "/")
    configure_hashiru(client)

    predictions: list[dict[str, Any]] = []

    for i, inst in enumerate(rows, 1):
        instance_id = inst.get("instance_id") or inst.get("instance") or inst.get("id")
        if not instance_id:
            print("Skipping instance without instance_id.", file=sys.stderr)
            continue

        problem_statement = inst.get("problem_statement") or ""
        hints_text = inst.get("hints_text") if "hints_text" in inst else None

        prompt = build_swebench_prompt(problem_statement=problem_statement, hints_text=hints_text)

        print(f"[{i}/{len(rows)}] Generating patch for {instance_id} ...")
        t0 = time.time()
        resp = client.predict({"text": prompt, "files": []}, None, api_name="/chat")
        # resp is typically (response, history) from gradio_client.
        if isinstance(resp, tuple) and len(resp) == 2:
            _response, history = resp
        else:
            history = resp

        # Extract last assistant text from history.
        model_text = ""
        if isinstance(history, list):
            for turn in reversed(history):
                if isinstance(turn, dict) and turn.get("role") == "assistant":
                    c = turn.get("content")
                    if isinstance(c, str) and c.strip():
                        model_text = c
                        break
        if not model_text and isinstance(resp, tuple):
            # Some Gradio setups return content in first element.
            maybe = resp[0]
            if isinstance(maybe, str):
                model_text = maybe

        patch = extract_patch_diff(model_text)
        elapsed = time.time() - t0
        print(f"  -> extracted patch chars={len(patch)} in {elapsed:.1f}s")

        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name_or_path,
                "model_patch": patch,
            }
        )

    # Write JSONL predictions.
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote predictions to {pred_path}")

    if args.write_only:
        return

    # Run official harness.
    instance_ids_for_harness = [r.get("instance_id") for r in rows if r.get("instance_id")]
    cmd = [
        args.swebench_python,
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        args.dataset_name,
        "-s",
        args.split,
        "-p",
        str(pred_path),
        "-i",
        *instance_ids_for_harness,
        "--max_workers",
        str(args.max_workers),
        "--timeout",
        str(args.timeout),
        "--run_id",
        run_id,
    ]
    print("Running SWE-bench official evaluation:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

