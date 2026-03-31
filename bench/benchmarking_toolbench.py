#!/usr/bin/env python3
"""
ToolBench-style benchmark driver for HASHIRU (Gradio).

This script does **not** import ToolBench's Python stack (gradio 3 / pydantic 1 / etc.).
It only reads ToolBench **instruction JSON** files from a cloned ToolBench tree (e.g.
`data/test_instruction/*.json`) and sends each `query` to HASHIRU via `gradio_client`.

**Venv layout (recommended):**
  - **HASHIRU venv:** run `python app.py --no-auth` (Gradio on 127.0.0.1:7860).
  - **This script:** any venv with `gradio_client` (+ optional `python-dotenv`), e.g.
        pip install gradio_client python-dotenv
    Run from `HASHIRU_Bench/bench/`:
        python benchmarking_toolbench.py --toolbench-root ./ToolBench
  - **ToolBench venv:** only needed if you run official **ToolEval** scoring later
    (see README_TOOLBENCH.md). No need to activate it to collect HASHIRU responses.

Results layout (aligned with BFCL / tau2 / IFBench style):
  HASHIRU_Bench/results/toolbench/toolbench_<timestamp>/
    meta.json                 # run configuration
    result/
      toolbench_hashiru.jsonl # one JSON object per query
      summary.json            # aggregates + optional per-file stats
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import zip_longest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# -----------------------------------------------------------------------------
# Optional .env (OPENAI_API_KEY etc. — only if you add custom eval hooks later)
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _bench_dir_for_dotenv = Path(__file__).resolve().parent
    load_dotenv(_bench_dir_for_dotenv / ".env")
    load_dotenv()
except ImportError:
    pass

_bench_dir = Path(__file__).resolve().parent
if str(_bench_dir) not in sys.path:
    sys.path.insert(0, str(_bench_dir))

from benchmarking_ifbench import (  # noqa: E402
    configure_hashiru_modes,
    get_last_assistant_content,
    looks_like_tool_plan,
)

# -----------------------------------------------------------------------------


TOOLBENCH_CEO_PREFIX = (
    "IMPORTANT CEO INSTRUCTIONS (ToolBench-style task):\n"
    "- The user instruction may require external tools or APIs. Use MemoryManager, "
    "ListFiles, ReadFile, GoogleSearchTool, GetWebsiteTool, or other loaded tools as needed.\n"
    "- Prefer existing agents via GetAgents + AskAgent when deep reasoning is required; "
    "avoid unnecessary AgentCreator loops.\n"
    "- When you have a final user-facing answer, output it clearly as plain text "
    "(no hidden tool JSON only).\n"
    "---\n\n"
)

TOOLBENCH_OUTPUT_SUFFIX = (
    "\n\nFINAL OUTPUT:\n"
    "Respond with the complete answer or plan the user would expect. "
    "If you used tools, summarize outcomes in natural language."
)


def build_hashiru_prompt_for_toolbench(user_query: str) -> str:
    q = (user_query or "").strip()
    return TOOLBENCH_CEO_PREFIX + q + TOOLBENCH_OUTPUT_SUFFIX


def load_instruction_json(path: Path) -> list[dict[str, Any]]:
    """Load ToolBench test_instruction JSON: list of {query, query_id, api_list, ...}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some dumps nest under a key
        for k in ("data", "instances", "items"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            raise ValueError(f"{path}: expected a JSON list or dict with a list field")
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be a list")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        q = row.get("query")
        if not q and row.get("instruction"):
            q = row.get("instruction")
        if not q:
            continue
        qid = row.get("query_id", row.get("id", f"{path.stem}_{i}"))
        out.append(
            {
                "query": str(q),
                "query_id": qid,
                "api_list": row.get("api_list", []),
                "_raw": row,
            }
        )
    return out


def discover_default_instruction_files(toolbench_root: Path) -> list[Path]:
    """Standard ToolBench locations after `data.zip` extract."""
    candidates = [
        toolbench_root / "data" / "test_instruction",
        toolbench_root / "ToolBench" / "data" / "test_instruction",
    ]
    files: list[Path] = []
    for d in candidates:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.is_file():
                files.append(p)
        break
    return files


def _maybe_json_load(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def load_queries_from_hf(
    dataset_id: str,
    dataset_config: str,
    splits: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Load ToolBench benchmark rows from Hugging Face datasets.
    Expected schema per row: query_id, query, api_list (often JSON string).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: datasets. Install with: pip install datasets"
        ) from e

    per_split: list[list[tuple[str, dict[str, Any]]]] = []
    for split in splits:
        ds = load_dataset(dataset_id, dataset_config, split=split)
        rows_for_split: list[tuple[str, dict[str, Any]]] = []
        for idx, row in enumerate(ds):
            if not isinstance(row, dict):
                continue
            q = row.get("query") or row.get("instruction")
            if not q:
                continue
            qid = row.get("query_id", row.get("id", f"{split}_{idx}"))
            api_list = _maybe_json_load(row.get("api_list", []))
            if not isinstance(api_list, list):
                api_list = []
            rows_for_split.append(
                (
                    f"hf:{split}",
                    {
                        "query": str(q),
                        "query_id": qid,
                        "api_list": api_list,
                        "_raw": row,
                    },
                )
            )
        per_split.append(rows_for_split)

    # Round-robin so --max-queries with multiple splits mixes task types instead of
    # exhausting the first split only.
    out: list[tuple[str, dict[str, Any]]] = []
    for batch in zip_longest(*per_split):
        for item in batch:
            if item is not None:
                out.append(item)
    return out


def _normalize_test_set_name(source_file: str) -> str:
    """
    Convert source identifiers to ToolEval test set names.
    Examples:
      - hf:g1_instruction -> G1_instruction
      - /path/G1_instruction.json -> G1_instruction
    """
    if source_file.startswith("hf:"):
        base = source_file.split(":", 1)[1]
    else:
        base = Path(source_file).stem
    m = base.lower()
    mapping = {
        "g1_instruction": "G1_instruction",
        "g1_category": "G1_category",
        "g1_tool": "G1_tool",
        "g2_instruction": "G2_instruction",
        "g2_category": "G2_category",
        "g3_instruction": "G3_instruction",
    }
    return mapping.get(m, base)


def _tooleval_available_tools(api_list: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(api_list, list):
        return out
    for i, api in enumerate(api_list):
        if not isinstance(api, dict):
            continue
        out.append(
            {
                "name": api.get("api_name", f"tool_{i}"),
                "description": api.get("api_description", ""),
                "parameters": {
                    "required": api.get("required_parameters", []),
                    "optional": api.get("optional_parameters", []),
                },
            }
        )
    return out


def _build_tooleval_answer_record(
    query: str,
    api_list: Any,
    final_answer: str,
) -> dict[str, Any]:
    """
    Build ToolEval-compatible converted answer entry for one query.
    We create a minimal chain ending with a Finish tool call so eval_pass_rate
    recognizes a final step.
    """
    finish_tool_message = {
        "name": "Finish",
        "arguments": {
            "return_type": "give_answer",
            "final_answer": final_answer,
        },
        "response": "",
    }
    answer_details = [
        {
            "role": "system",
            "message": "ToolBench adapter execution graph",
            "next": [
                {
                    "role": "user",
                    "message": query,
                    "next": [
                        {
                            "role": "assistant",
                            "message": "Generate final answer from HASHIRU.",
                            "next": [
                                {
                                    "role": "tool",
                                    "message": finish_tool_message,
                                    "next": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    return {
        "query": query,
        "available_tools": _tooleval_available_tools(api_list),
        "answer": {
            "method": "HASHIRU",
            "total_steps": 4,
            "final_answer": final_answer,
            "answer_details": answer_details,
        },
    }


def _prepare_tooleval_api_pool_file(out_root: Path) -> Path:
    """
    Ensure ToolEval gets a valid API pool file instead of the placeholder path in
    evaluator config. We derive credentials from env vars.
    """
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "ToolEval pass-rate requires an OpenAI key. Set OPENAI_API_KEY (or OPENAI_KEY) "
            "before running with --run-tooleval-passrate."
        )
    pool_item = {
        "username": "hashiru_tooleval",
        "passwd": "unused",
        "api_key": api_key,
    }
    org = os.environ.get("OPENAI_ORG") or os.environ.get("OPENAI_ORGANIZATION")
    if org:
        pool_item["organization"] = org
    api_pool_file = out_root / "tooleval" / "api_pool.json"
    api_pool_file.parent.mkdir(parents=True, exist_ok=True)
    api_pool_file.write_text(
        json.dumps([pool_item], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return api_pool_file


def _choose_tooleval_python(explicit_python: str | None) -> str:
    """
    Pick a python interpreter for ToolEval.
    Priority:
      1) --tooleval-python
      2) ./ToolBench/.venv/bin/python (if exists)
      3) current interpreter

    Use .absolute() not .resolve(): venv ``bin/python`` is often a symlink to
    e.g. /usr/bin/python3.x; resolving would replace the path with the system
    binary and drop the venv's site-packages.
    """
    if explicit_python:
        return str(Path(explicit_python).expanduser().absolute())
    candidate = _bench_dir / "ToolBench" / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate.absolute())
    return sys.executable


def _openai_major_for_python(python_exec: str) -> int | None:
    """
    Return major version of openai in the given interpreter, or None if unknown.
    """
    cmd = [
        python_exec,
        "-c",
        "import openai; v=getattr(openai,'__version__','0.0.0'); print(v)",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        v = (proc.stdout or "").strip()
        if not v:
            return None
        return int(v.split(".", 1)[0])
    except Exception:
        return None


def _missing_modules_for_python(python_exec: str, modules: list[str]) -> list[str]:
    """
    Return module names that cannot be imported in the target interpreter.
    """
    missing: list[str] = []
    for m in modules:
        cmd = [python_exec, "-c", f"import {m}"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except Exception:
            missing.append(m)
    return missing


def configure_hashiru_modes_empty(client: Any) -> None:
    """BFCL-style: no mode indices → budgets/agents off in default HASHIRU build."""
    try:
        client.predict(modeIndexes=[], api_name="/update_model")
    except Exception:
        pass


def _relative_or_name(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def iter_queries_from_files(paths: list[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    for p in paths:
        try:
            rows = load_instruction_json(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}", file=sys.stderr)
            continue
        for row in rows:
            yield str(p), row


def run_chat_turn(
    client: Any,
    text: str,
    history: Any,
) -> tuple[Any, Any]:
    resp = client.predict(
        {"text": text, "files": []},
        history,
        api_name="/chat",
    )
    if isinstance(resp, tuple) and len(resp) == 2:
        _r, history = resp
    else:
        history = resp
    return history, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ToolBench instruction files against HASHIRU (Gradio).",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["hf", "local"],
        default="hf",
        help="Load instructions from Hugging Face (default) or local JSON files.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="tuandunghcmut/toolbench-v1",
        help="Hugging Face dataset id for ToolBench benchmark data.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="benchmark",
        help="Hugging Face dataset configuration (default: benchmark).",
    )
    parser.add_argument(
        "--hf-splits",
        nargs="*",
        default=[
            "g1_instruction",
            "g1_category",
            "g1_tool",
            "g2_instruction",
            "g2_category",
            "g3_instruction",
        ],
        help="Benchmark splits to pull from Hugging Face.",
    )
    parser.add_argument(
        "--toolbench-root",
        type=str,
        default=None,
        help=(
            "Path to cloned ToolBench repo (must contain data/test_instruction after data.zip). "
            "Default: ./ToolBench next to this script if it exists."
        ),
    )
    parser.add_argument(
        "--instruction-files",
        nargs="*",
        default=None,
        help=(
            "Explicit JSON files (ToolBench test_instruction format). "
            "If omitted, uses all *.json under <toolbench-root>/data/test_instruction/."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Stop after this many queries (across all files).",
    )
    parser.add_argument(
        "--gradio-url",
        type=str,
        default=os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860"),
        help="HASHIRU Gradio base URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Result root (default: HASHIRU_Bench/results/toolbench/toolbench_<timestamp>).",
    )
    parser.add_argument(
        "--empty-modes",
        action="store_true",
        help="Set Gradio modeIndexes=[] (minimal CEO/tools — faster smoke tests).",
    )
    parser.add_argument(
        "--max-continue-rounds",
        type=int,
        default=3,
        help="If HASHIRU returns internal tool-plan JSON, send follow-ups this many times.",
    )
    parser.add_argument(
        "--toolbench-python",
        type=str,
        default=None,
        help=(
            "Path to python executable in the ToolBench venv — printed in meta.json for "
            "running official ToolEval later (this script does not invoke ToolEval)."
        ),
    )
    parser.add_argument(
        "--emit-tooleval-format",
        action="store_true",
        help="Write ToolEval converted-answer and test_id artifacts.",
    )
    parser.add_argument(
        "--run-tooleval-passrate",
        action="store_true",
        help="After emitting ToolEval artifacts, run ToolEval eval_pass_rate.py.",
    )
    parser.add_argument(
        "--tooleval-evaluator",
        type=str,
        default="tooleval_gpt-3.5-turbo_default",
        help="Evaluator id used by ToolEval pass-rate script.",
    )
    parser.add_argument(
        "--tooleval-max-eval-threads",
        type=int,
        default=10,
        help="Max ToolEval threads for pass-rate run.",
    )
    parser.add_argument(
        "--tooleval-evaluate-times",
        type=int,
        default=2,
        help="How many evaluation repeats per query in ToolEval.",
    )
    parser.add_argument(
        "--tooleval-python",
        type=str,
        default=None,
        help=(
            "Python interpreter for ToolEval subprocess. Must have tenacity, "
            "pyyaml, tqdm, and openai installed."
        ),
    )
    args = parser.parse_args()

    default_root = _bench_dir / "ToolBench"
    toolbench_root = Path(args.toolbench_root or default_root).expanduser().resolve()

    files: list[Path] = []
    hf_rows: list[tuple[str, dict[str, Any]]] = []

    if args.dataset_source == "hf":
        if not args.hf_splits:
            raise SystemExit("--hf-splits cannot be empty when --dataset-source=hf")
        hf_rows = load_queries_from_hf(args.hf_dataset, args.hf_config, args.hf_splits)
        if not hf_rows:
            raise SystemExit(
                f"No rows loaded from Hugging Face dataset {args.hf_dataset} "
                f"(config={args.hf_config}, splits={args.hf_splits})."
            )
    else:
        if args.toolbench_root is None and not toolbench_root.is_dir():
            raise SystemExit(
                "Set --toolbench-root to your cloned ToolBench repo (with data/test_instruction). "
                f"Expected default {toolbench_root} missing."
            )
        if not toolbench_root.is_dir():
            raise SystemExit(f"--toolbench-root is not a directory: {toolbench_root}")

        if args.instruction_files:
            files = [Path(f).expanduser().resolve() for f in args.instruction_files]
            for f in files:
                if not f.is_file():
                    raise SystemExit(f"Instruction file not found: {f}")
        else:
            files = discover_default_instruction_files(toolbench_root)
            if not files:
                raise SystemExit(
                    f"No instruction JSON files found under {toolbench_root}/data/test_instruction. "
                    "Download ToolBench data.zip (see ToolBench README) and extract, or pass "
                    "--instruction-files explicitly."
                )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        out_root = (
            _bench_dir.parent / "results" / "toolbench" / f"toolbench_{ts}"
        ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    result_dir = out_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = result_dir / "toolbench_hashiru.jsonl"

    try:
        from gradio_client import Client
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: gradio_client. Install with: pip install gradio_client"
        ) from e

    base = args.gradio_url.rstrip("/") + "/"
    try:
        client = Client(base)
    except Exception as e:
        raise SystemExit(f"Failed to connect to HASHIRU Gradio at {base}: {e}") from e

    if args.empty_modes:
        configure_hashiru_modes_empty(client)
    else:
        configure_hashiru_modes(client)

    meta = {
        "benchmark": "toolbench_hashiru",
        "schema_version": 1,
        "run_timestamp_utc": ts,
        "hashiru_gradio_url": base,
        "dataset_source": args.dataset_source,
        "toolbench_root": str(toolbench_root) if args.dataset_source == "local" else None,
        "instruction_files": [str(f) for f in files],
        "hf_dataset": args.hf_dataset if args.dataset_source == "hf" else None,
        "hf_config": args.hf_config if args.dataset_source == "hf" else None,
        "hf_splits": args.hf_splits if args.dataset_source == "hf" else None,
        "empty_modes": bool(args.empty_modes),
        "max_continue_rounds": args.max_continue_rounds,
        "toolbench_python_hint": args.toolbench_python,
        "note": (
            "Responses are HASHIRU chat outputs, not ToolBench qa_pipeline trajectories. "
            "Official ToolEval pass/win rates require converting to ToolBench prediction format; "
            "see README_TOOLBENCH.md."
        ),
    }
    (out_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"ToolBench → HASHIRU: writing {jsonl_path}")
    if args.dataset_source == "hf":
        print(
            f"  hf: {args.hf_dataset} [{args.hf_config}] splits={args.hf_splits} "
            f"rows={len(hf_rows)} empty_modes={args.empty_modes}"
        )
    else:
        print(f"  files: {len(files)}  empty_modes={args.empty_modes}")

    per_file_counts: dict[str, dict[str, int]] = {}
    tooleval_by_split: dict[str, dict[str, Any]] = {}
    tooleval_ids_by_split: dict[str, dict[str, int]] = {}
    total_elapsed = 0.0
    n_ok = 0
    n_err = 0
    n_written = 0

    row_iter: Iterator[tuple[str, dict[str, Any]]]
    if args.dataset_source == "hf":
        row_iter = iter(hf_rows)
    else:
        row_iter = iter_queries_from_files(files)

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for src_path, row in row_iter:
            if args.max_queries is not None and n_written >= args.max_queries:
                break

            key_file = (
                src_path if args.dataset_source == "hf"
                else _relative_or_name(toolbench_root, Path(src_path))
            )
            per_file_counts.setdefault(key_file, {"n": 0, "errors": 0})

            user_q = row["query"]
            prompt = build_hashiru_prompt_for_toolbench(user_q)
            err: str | None = None
            response_text = ""
            t0 = time.perf_counter()
            try:
                history = None
                history, history = run_chat_turn(client, prompt, history)
                response_text = get_last_assistant_content(history)
                rounds = 0
                while (
                    (not response_text or looks_like_tool_plan(response_text))
                    and rounds < max(0, args.max_continue_rounds)
                ):
                    rounds += 1
                    history, history = run_chat_turn(
                        client,
                        "Provide only the final natural-language answer for the user. "
                        "Do not output agent orchestration JSON.",
                        history,
                    )
                    response_text = get_last_assistant_content(history)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
                per_file_counts[key_file]["errors"] += 1
            else:
                n_ok += 1
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            per_file_counts[key_file]["n"] += 1

            if isinstance(response_text, str):
                response_text = response_text.strip()

            record = {
                "source_file": key_file,
                "query_id": row["query_id"],
                "query": user_q,
                "api_list": row.get("api_list", []),
                "hashiru_response": response_text,
                "elapsed_sec": round(elapsed, 3),
                "error": err,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Build ToolEval-compatible converted answer entries.
            split_name = _normalize_test_set_name(key_file)
            qid = str(row["query_id"])
            tooleval_by_split.setdefault(split_name, {})
            tooleval_ids_by_split.setdefault(split_name, {})
            tooleval_by_split[split_name][qid] = _build_tooleval_answer_record(
                query=user_q,
                api_list=row.get("api_list", []),
                final_answer=response_text,
            )
            tooleval_ids_by_split[split_name][qid] = 1

            n_written += 1

            if n_written % 5 == 0 or err:
                flag = "ERR" if err else "ok"
                print(f"  [{n_written}] {flag} id={row['query_id']} {elapsed:.1f}s")

    summary = {
        "benchmark": "toolbench_hashiru",
        "run_timestamp_utc": ts,
        "total_queries": n_written,
        "completed_without_exception": n_ok,
        "exceptions": n_err,
        "total_elapsed_sec": round(total_elapsed, 3),
        "per_file": per_file_counts,
        "result_jsonl": _relative_or_name(out_root, jsonl_path),
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.emit_tooleval_format or args.run_tooleval_passrate:
        tooleval_root = out_root / "tooleval"
        converted_model_dir = tooleval_root / "converted" / "hashiru_toolbench"
        test_ids_dir = tooleval_root / "test_ids"
        passrate_out_dir = tooleval_root / "pass_rate_results"
        converted_model_dir.mkdir(parents=True, exist_ok=True)
        test_ids_dir.mkdir(parents=True, exist_ok=True)
        passrate_out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, items in tooleval_by_split.items():
            (converted_model_dir / f"{split_name}.json").write_text(
                json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        for split_name, ids in tooleval_ids_by_split.items():
            (test_ids_dir / f"{split_name}.json").write_text(
                json.dumps(ids, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        print(f"ToolEval converted artifacts → {tooleval_root}")

        if args.run_tooleval_passrate:
            tooleval_dir = _bench_dir / "ToolBench" / "toolbench" / "tooleval"
            if not tooleval_dir.is_dir():
                raise SystemExit(
                    f"ToolEval directory not found: {tooleval_dir}. "
                    "Clone ToolBench under HASHIRU_Bench/bench/ToolBench first."
                )
            tooleval_python = _choose_tooleval_python(args.tooleval_python)
            required_modules = ["tenacity", "yaml", "tqdm", "openai"]
            missing_mods = _missing_modules_for_python(tooleval_python, required_modules)
            if missing_mods:
                raise SystemExit(
                    "Selected ToolEval interpreter is missing required modules: "
                    f"{', '.join(missing_mods)}\n"
                    f"Interpreter: {tooleval_python}\n"
                    "Install them in that env, e.g.:\n"
                    f"  {tooleval_python} -m pip install tenacity pyyaml tqdm openai"
                )
            major = _openai_major_for_python(tooleval_python)
            print(f"ToolEval python: {tooleval_python} (openai major={major})")

            cmd = [
                tooleval_python,
                "eval_pass_rate.py",
                "--converted_answer_path",
                str(tooleval_root / "converted"),
                "--save_path",
                str(passrate_out_dir),
                "--reference_model",
                "hashiru_toolbench",
                "--test_ids",
                str(test_ids_dir),
                "--evaluator",
                args.tooleval_evaluator,
                "--max_eval_threads",
                str(args.tooleval_max_eval_threads),
                "--evaluate_times",
                str(args.tooleval_evaluate_times),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(tooleval_dir)
            api_pool_file = _prepare_tooleval_api_pool_file(out_root)
            # ToolEval's OpenaiPoolRequest reads API_POOL_FILE and OPENAI_KEY.
            env["API_POOL_FILE"] = str(api_pool_file)
            if "OPENAI_KEY" not in env and env.get("OPENAI_API_KEY"):
                env["OPENAI_KEY"] = env["OPENAI_API_KEY"]
            print("Running ToolEval pass-rate:")
            print("  " + " ".join(cmd))
            subprocess.run(cmd, cwd=str(tooleval_dir), env=env, check=True)

    print(f"Done. {n_written} rows → {jsonl_path}")
    print(f"Summary → {result_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
