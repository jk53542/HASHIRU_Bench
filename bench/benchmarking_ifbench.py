#!/usr/bin/env python3
"""
IFBench runner using HASHIRU.

IFBench evaluation expects:
1) input_data: IFBench_test.jsonl
2) input_response_data: JSONL where each line is:
     {"prompt": <original prompt from input_data>, "response": <model response>}

This script:
 - loads input_data
 - calls HASHIRU for each prompt with:
   - budgets OFF
   - tool invocation ON
   - CEO prefix containing "You MUST use agents..."
 - writes input_response_data
 - runs IFBench official evaluation:
     python -m run_eval --input_data=... --input_response_data=... --output_dir=...

Notes:
 - We MUST store responses using the ORIGINAL `prompt` string from IFBench_test.jsonl,
   because evaluation_lib maps prompt->response by exact string equality.
 - We may modify the prompt sent to HASHIRU, but the JSONL key remains the original prompt.
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
    "- You MUST delegate reasoning to one or more agents.\n"
    "- After agents respond, output ONLY the final answer to the user prompt.\n"
)


def get_last_assistant_content(history: Any) -> str:
    """
    Extract the last assistant message content from gradio_client chat history.
    Compatible with both:
      - list[dict] history: {"role": "assistant", "content": "..."}
      - legacy list[(user, bot)] history
    """
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return ""

    # Legacy gradio: list of (user, bot) tuples
    if history and isinstance(history[-1], (list, tuple)) and len(history[-1]) >= 2:
        last = history[-1]
        return str(last[1] or "")

    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            return content
        fr = turn.get("function_response") or {}
        out = (fr.get("result") or {}).get("output")
        if out:
            return str(out)
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts") or []
            if parts and isinstance(parts[0], dict) and parts[0].get("text"):
                return str(parts[0]["text"])
    return ""


def looks_like_tool_plan(text: str) -> bool:
    """
    Detect HASHIRU internal tool-plan outputs, e.g. JSON blocks like:
      [{"agent_name": "...", "prompt": "..."}]
    Those are not valid IFBench responses.
    """
    if not text:
        return False
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n```$", "", t).strip()
    if ("\"agent_name\"" in t and "\"prompt\"" in t) or ("'agent_name'" in t and "'prompt'" in t):
        return True
    return False


def configure_hashiru_modes(client: Any) -> None:
    """
    Configure HASHIRU modes:
      - budgets OFF
      - tool invocation ON
    We attempt mode names first, then fall back to mode indices (for compatibility).
    """
    mode_names = [
        "ENABLE_AGENT_CREATION",
        "ENABLE_LOCAL_AGENTS",
        "ENABLE_TOOL_CREATION",
        "ENABLE_TOOL_INVOCATION",
        "ENABLE_MEMORY",
    ]
    mode_indices = [0, 1, 3, 4, 7]  # matches paper-review safe set (no budget modes)

    # Update model config once.
    try:
        client.predict(modeIndexes=mode_names, api_name="/update_model")
        return
    except Exception:
        pass

    client.predict(modeIndexes=mode_indices, api_name="/update_model")


def build_hashiru_prompt(original_prompt: str) -> str:
    """
    Build the prompt sent to HASHIRU.
    We include CEO FORCE AGENTS prefix, but we keep the output constraint: "final answer only".
    """
    original_prompt = (original_prompt or "").strip()
    return (
        CEO_FORCE_AGENTS_PREFIX
        + "\n"
        + original_prompt
        + "\n\nFINAL OUTPUT CONSTRAINT:\n"
        + "Return only the response text that satisfies the instructions above. "
        + "Do not include extra headings or analysis."
    )


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="IFBench evaluation runner using HASHIRU.")
    parser.add_argument(
        "--ifbench-root",
        type=str,
        required=True,
        help="Path to the cloned IFBench repo (where run_eval.py lives).",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=None,
        help="Path to IFBench_test.jsonl (default: <ifbench-root>/data/IFBench_test.jsonl).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit how many prompts to run (useful for smoke tests).",
    )
    parser.add_argument(
        "--gradio-url",
        type=str,
        default=os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860"),
        help="HASHIRU Gradio base URL.",
    )
    parser.add_argument(
        "--responses-out",
        type=str,
        default=None,
        help="Optional path for the generated input_response_data JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="IFBench evaluation output directory (default: <ifbench-root>/eval/<timestamp>).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only generate responses JSONL; do not run IFBench run_eval.",
    )
    parser.add_argument(
        "--max-continue-rounds",
        type=int,
        default=4,
        help="If HASHIRU returns internal tool-plan JSON instead of a final answer, continue chat this many times to force final output.",
    )
    args = parser.parse_args()

    ifbench_root = Path(args.ifbench_root).expanduser().resolve()
    if not (ifbench_root / "run_eval.py").is_file():
        raise SystemExit(f"Could not find IFBench run_eval.py under: {ifbench_root}")

    input_data = (
        Path(args.input_data).expanduser().resolve()
        if args.input_data
        else (ifbench_root / "data" / "IFBench_test.jsonl")
    )
    if not input_data.is_file():
        raise SystemExit(f"IFBench input data not found: {input_data}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (ifbench_root / "eval" / f"hashiru_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_out = (
        Path(args.responses_out).expanduser().resolve()
        if args.responses_out
        else (output_dir / f"hashiru-responses.jsonl")
    )

    try:
        from gradio_client import Client
    except ImportError as e:
        raise SystemExit("Missing dependency: gradio_client. Install with: pip install gradio_client") from e

    try:
        # Instantiate client
        client = Client(args.gradio_url.rstrip("/") + "/")
    except Exception as e:
        raise SystemExit(f"Failed to init gradio client: {e}") from e

    configure_hashiru_modes(client)

    # Load prompts and generate responses.
    rows = list(iter_jsonl(input_data))
    if not rows:
        raise SystemExit("IFBench input data is empty.")
    if args.max_prompts is not None and args.max_prompts > 0:
        rows = rows[: args.max_prompts]

    print(f"IFBench: loaded {len(rows)} prompts from {input_data}")
    print(f"Writing responses to {responses_out}")

    with open(responses_out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(rows, 1):
            key = ex.get("key")
            original_prompt = ex.get("prompt") or ""
            if not original_prompt:
                # Still write a placeholder so evaluation won't crash.
                f.write(json.dumps({"prompt": original_prompt, "response": ""}) + "\n")
                continue

            hashiru_prompt = build_hashiru_prompt(original_prompt)

            start = time.time()
            resp = client.predict(
                {"text": hashiru_prompt, "files": []},
                None,
                api_name="/chat",
            )
            if isinstance(resp, tuple) and len(resp) == 2:
                _response, history = resp
            else:
                history = resp

            model_resp = get_last_assistant_content(history)
            rounds = 0
            while (not model_resp or looks_like_tool_plan(model_resp)) and rounds < max(0, args.max_continue_rounds):
                rounds += 1
                followup = (
                    "Return ONLY the final user-facing answer to the original request. "
                    "Do NOT output tool plans, JSON arrays, agent_name fields, or internal orchestration details."
                )
                resp = client.predict(
                    {"text": followup, "files": []},
                    history,
                    api_name="/chat",
                )
                if isinstance(resp, tuple) and len(resp) == 2:
                    _response, history = resp
                else:
                    history = resp
                model_resp = get_last_assistant_content(history)

            elapsed = time.time() - start

            # Defensive cleanup: remove obvious surrounding whitespace only.
            if isinstance(model_resp, str):
                model_resp = model_resp.strip()
            else:
                model_resp = str(model_resp)

            f.write(
                json.dumps(
                    {
                        "prompt": original_prompt,
                        "response": model_resp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if i % 10 == 0 or i == len(rows):
                plan_flag = "tool-plan" if looks_like_tool_plan(model_resp) else "final"
                print(f"  [{i}/{len(rows)}] key={key} elapsed={elapsed:.1f}s rounds={rounds} type={plan_flag}")

    print(f"Generated responses JSONL: {responses_out}")

    if args.skip_eval:
        return

    # Run official evaluation (uses absl).
    # IMPORTANT: IFBench run_eval expects prompt_to_response to contain keys for every
    # prompt in input_data. If we benchmark a subset (e.g. --max-prompts), we must also
    # evaluate against the same subset input file to avoid KeyError.
    eval_input_data = input_data
    try:
        # rows is the exact prompt subset we generated responses for.
        if len(rows) != sum(1 for _ in iter_jsonl(input_data)):
            eval_input_data = output_dir / "ifbench_input_subset.jsonl"
            write_jsonl(eval_input_data, rows)
    except Exception:
        # Fallback: if counting failed for any reason, still use original input_data.
        eval_input_data = input_data

    cmd = [
        sys.executable,
        "-m",
        "run_eval",
        f"--input_data={str(eval_input_data)}",
        f"--input_response_data={str(responses_out)}",
        f"--output_dir={str(output_dir)}",
    ]
    env = os.environ.copy()
    # Ensure IFBench root is on PYTHONPATH so run_eval can import evaluation_lib, instructions_registry, etc.
    env["PYTHONPATH"] = str(ifbench_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print("Running IFBench official evaluation:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ifbench_root), env=env, check=True)


if __name__ == "__main__":
    main()

