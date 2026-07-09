"""
GSM8K benchmark via HASHIRU Gradio (or Gemini Flash baseline).

Run from the ``bench/`` directory (same as MMLU) so imports resolve::

    cd HASHIRU_Bench/bench
    python benchmarking_GSM8K.py -m hashiru --num_samples 5 --offset 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from datasets import load_dataset
from gradio_client import Client
from google import genai
from google.genai import types
from tqdm import tqdm

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_DELEGATE_THEN_TASK
from benchmark_trace_context import hashiru_trace_context_prefix

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
args = None  # set in main; used by get_client / call_api (matches MMLU pattern)


def get_client():
    if args.model_name in ["hashiru"]:
        client = Client("http://127.0.0.1:7860/")
        client.predict(
            modeIndexes=[
                "ENABLE_AGENT_CREATION",
                "ENABLE_LOCAL_AGENTS",
                "ENABLE_CLOUD_AGENTS",
                "ENABLE_TOOL_CREATION",
                "ENABLE_TOOL_INVOCATION",
                "ENABLE_RESOURCE_BUDGET",
                "ENABLE_ECONOMY_BUDGET",
            ],
            api_name="/update_model",
        )
        return client
    if args.model_name in ["flash2.0"]:
        return genai.Client(api_key=API_KEY)
    raise ValueError(f"Unknown model_name: {args.model_name}")


def _get_last_assistant_content(resp):
    """Best-effort extraction of final assistant text from Gradio history."""
    if isinstance(resp, tuple):
        resp = resp[0]
    if not isinstance(resp, list):
        return ""
    for turn in reversed(resp):
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        c = turn.get("content")
        if isinstance(c, str) and c:
            return c
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return str(out)
    return ""


def call_api(client, instruction, inputs, tries=0, trace_prefix="", prompt_body=""):
    start = time.time()
    if args.model_name == "hashiru":
        if tries > 3:
            print("Error: too many tries")
            return ""
        client = Client("http://127.0.0.1:7860/")
        client.predict(
            modeIndexes=[
                "ENABLE_AGENT_CREATION",
                "ENABLE_LOCAL_AGENTS",
                "ENABLE_CLOUD_AGENTS",
                "ENABLE_TOOL_CREATION",
                "ENABLE_TOOL_INVOCATION",
                "ENABLE_RESOURCE_BUDGET",
                "ENABLE_ECONOMY_BUDGET",
            ],
            api_name="/update_model",
        )
        message_text = (trace_prefix + (prompt_body or (instruction + inputs))).strip()
        response, history = client.predict(
            message={"text": message_text, "files": []},
            api_name="/chat",
        )
        if "error" in response.get("content", ""):
            time.sleep(60)
            return call_api(
                client,
                instruction,
                inputs,
                tries + 1,
                trace_prefix=trace_prefix,
                prompt_body=prompt_body,
            )
        print("cost time", time.time() - start)
        return _get_last_assistant_content(history) or response.get("content", "")
    if args.model_name == "flash2.0":
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        body = prompt_body or (instruction + inputs)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=body,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    safety_settings=safety_settings,
                ),
            )
        except Exception:
            if tries > 3:
                print("Error: too many tries")
                return ""
            time.sleep(60)
            return call_api(
                client, instruction, inputs, tries + 1,
                trace_prefix=trace_prefix, prompt_body=prompt_body,
            )
        print("cost time", time.time() - start)
        return response.text or ""
    return ""


def _build_task_prompt(question: str, require_agent_delegation: bool) -> str:
    if require_agent_delegation:
        return (
            "You must solve this math question by delegating to an agent. "
            "Do NOT use Python, calculator, or other math tools yourself. "
            "Create or use an existing agent that is a math expert, send the question to that agent, "
            "and report back only the final numerical answer. "
            f"Question: {question} "
            "Reply with the answer in the format: <ANSWER> (number only, no explanation)."
        )
    return (
        "Help: You are a student agent taking a math test. "
        "Feel free to use tools or agents to help you answer the question. "
        "You can decide what tools or agents to use. "
        "Your goal is to answer the question correctly. "
        f"You have been asked the following question: {question}\n"
        "Provide the answer in the format: <ANSWER> (numeric final answer). "
        "Prefer a short final line with just the number if possible."
    )


def benchmark_gsm8k():
    require_delegation = args.require_agent_delegation
    split = args.split.strip().lower()
    if split not in ("train", "test"):
        raise SystemExit("--split must be train or test")

    print(f"Loading GSM8K ({split})...")
    dataset = load_dataset("openai/gsm8k", "main")
    if split not in dataset:
        raise SystemExit(f"Split '{split}' not in dataset")
    rows = list(dataset[split])
    start_i = max(0, int(args.offset))
    end_i = len(rows) if args.num_samples is None else min(
        len(rows), start_i + max(0, int(args.num_samples))
    )
    selection = rows[start_i:end_i]
    if not selection:
        raise SystemExit(f"No rows in range offset={start_i}, end={end_i} (split size {len(rows)})")
    print(
        f"Evaluating GSM8K split={split} rows [{start_i}, {end_i}) "
        f"count={len(selection)} (split_size={len(rows)})"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_name
    if not out_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"gsm8k_{split}_{ts}.jsonl"
    out_path = os.path.join(args.output_dir, out_name)
    print(f"Writing results to {out_path}")

    client = get_client()
    results = []

    for k, sample in enumerate(tqdm(selection, desc="gsm8k")):
        bench_index = k + 1
        global_row_index = start_i + k
        question = sample["question"]
        answer = sample["answer"]
        answer_only = answer.split("####")[-1].strip()

        visible_task = _build_task_prompt(question, require_delegation)
        task_body = f"{CEO_FORCE_AGENTS_PREFIX_DELEGATE_THEN_TASK}\n{visible_task}"

        trace_prefix = ""
        if args.model_name == "hashiru":
            trace_prefix = hashiru_trace_context_prefix(
                benchmark_name="gsm8k",
                question_index=bench_index,
                question_id=f"{split}_{global_row_index}",
                bench_attempt=1,
                question_text=(question or "")[:2500],
            )

        t0 = time.time()
        agent_resp = call_api(
            client,
            "",
            visible_task,
            trace_prefix=trace_prefix,
            prompt_body=task_body,
        )
        elapsed = time.time() - t0

        # Match: reference uses substring check in model output
        normalized = (agent_resp or "").replace(",", "").strip()
        is_correct = answer_only in agent_resp or answer_only in normalized

        row = {
            "split": split,
            "row_index": global_row_index,
            "bench_index": bench_index,
            "question_id": f"{split}_{global_row_index}",
            "question": question,
            "answer": answer,
            "answer_only": answer_only,
            "agent_resp": agent_resp,
            "is_correct": is_correct,
            "time_elapsed": elapsed,
            "model_name": args.model_name,
        }
        results.append(row)
        if int(args.sleep_between) > 0:
            time.sleep(int(args.sleep_between))

    with open(out_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in results if r["is_correct"])
    print(f"Done. accuracy={n_ok}/{len(results)} written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GSM8K against HASHIRU (Gradio) or Gemini 2.0 Flash baseline.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="eval_results",
        help="Directory for the JSONL result file (default: eval_results).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Result filename (default: gsm8k_<split>_<timestamp>.jsonl).",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="hashiru",
        choices=["hashiru", "flash2.0"],
        help="hashiru = local Gradio (127.0.0.1:7860); flash2.0 = Gemini API (needs API key).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=None,
        help="Number of problems to run from --offset (default: run through end of split).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based index into the chosen split before taking --num_samples (default: 0).",
    )
    parser.add_argument(
        "--require_agent_delegation",
        action="store_true",
        help="Stricter prompt: delegate to a math agent; do not use Python/calculator as CEO.",
    )
    parser.add_argument(
        "--sleep_between",
        type=int,
        default=0,
        help="Seconds to sleep after each problem (default: 0; use to throttle rate limits).",
    )
    args = parser.parse_args()
    benchmark_gsm8k()
