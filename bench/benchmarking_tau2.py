#!/usr/bin/env python3
"""
τ²-bench (tau2-bench) benchmark for HASHIRU: runs tau2 simulations with HASHIRU as the agent.

Prerequisites:
  - HASHIRU running at http://127.0.0.1:7860/ (or set HASHIRU_GRADIO_URL / --gradio_url).
  - tau2-bench installed: pip install "git+https://github.com/sierra-research/tau2-bench@main"
  - Tau2 data directory: set TAU2_DATA_DIR (see "Tau2 data directory setup" below).
  - User simulator LLM: see "User simulator LLM setup" below.

Tau2 data directory setup
------------------------
Tau2 needs domain task files (e.g. data/tau2/domains/airline/tasks.json). With a normal
pip install these are not included, so you must set TAU2_DATA_DIR before running.

Option A - Clone the tau2-bench repo (recommended):
  1. Clone: git clone https://github.com/sierra-research/tau2-bench.git
  2. Set the env var to the repo's data folder (the one that contains the tau2 subfolder):
       export TAU2_DATA_DIR=/path/to/tau2-bench/data
  3. Run the benchmark from HASHIRU_Bench/bench (or set TAU2_DATA_DIR in .env).

Option B - Download only the data from Hugging Face:
  1. Install Hugging Face CLI: pip install huggingface_hub
  2. Create a directory and download:
       mkdir -p tau2_data && cd tau2_data
       huggingface-cli download HuggingFaceH4/tau2-bench-data --repo-type dataset --local-dir .
  3. The dataset may use a different layout; if so, ensure you have tau2/domains/<domain>/ under
     a single root and set TAU2_DATA_DIR to that root (so that TAU2_DATA_DIR/tau2/domains/airline/
     exists).

Verify: run tau2 check-data (with TAU2_DATA_DIR set); or run this script and check the error.

User simulator LLM setup
-----------------------
In τ²-bench there are two roles: the *agent* (your system under test) and the *user simulator*
(the simulated "customer"). We use HASHIRU as the agent. The *user simulator* is still an LLM
that tau2 calls to generate the customer's messages in each conversation. Tau2 uses LiteLLM
to call that LLM, so you must provide an API key for whatever model you pass as --user_llm.

Steps:

  1. Choose a model for the user simulator (e.g. gpt-4.1, which is common in tau2 docs).
     You will pass it as: --user_llm gpt-4.1

  2. Get an API key for that model's provider (e.g. OpenAI for gpt-4.1).

  3. Make the key available to the process. LiteLLM reads keys from environment variables:
     - For OpenAI (gpt-4.1, gpt-4o, etc.): set OPENAI_API_KEY
     - For other providers see: https://docs.litellm.ai/docs/providers

  4. Easiest: create a .env file with your API key. This script loads .env automatically
     if python-dotenv is installed (pip install python-dotenv). Place .env in the same
     directory as this file (HASHIRU_Bench/bench/) or in your current working directory.
     Example .env:

        OPENAI_API_KEY=sk-your-openai-key-here

     Then run:
         python benchmarking_tau2.py --domain airline --num_tasks 3 --user_llm gpt-4.1

     Without python-dotenv, set the variable in the shell before running:
         export OPENAI_API_KEY=sk-your-openai-key-here
         python benchmarking_tau2.py --domain airline --num_tasks 3 --user_llm gpt-4.1

  5. If you use a different model (e.g. from Anthropic), set the corresponding env var
     (e.g. ANTHROPIC_API_KEY) and pass --user_llm claude-3-5-sonnet-20241022 or similar.

Usage:
  python benchmarking_tau2.py --domain airline --num_tasks 3 --num_trials 1 --user_llm gpt-4.1
  python benchmarking_tau2.py --domain retail --num_tasks 5 --user_llm gpt-4.1 --gradio_url http://127.0.0.1:7860/

Verbose runs & cost (HASHIRU side)
-----------------------------------
Tau2 prepends CEO delegation text that triggers manager worker enforcement (AskAgent /
AskMultipleAgents). That is expected; occasional lines
``Worker mandate: CEO produced text without AskAgent/AskMultipleAgents`` mean the CEO
emitted plain text once before delegating in that segment—the manager adds one SYSTEM
nudge (up to HASHIRU_MANDATE_ASK_AGENT_MAX_NUDGES, default 4). That is usually a small
fraction of total work.

The *large* repetition in logs is usually **semantic metrics sampling**: each AskAgent
runs the worker once plus several extra stochastic samples for entropy/density, each
printed as ``Asked Agent ... answered with``. To run **one** worker call per AskAgent
(faster, quieter benchmarks), disable both metrics before starting app.py:

  export HASHIRU_ENABLE_SEMANTIC_ENTROPY=0
  export HASHIRU_ENABLE_SEMANTIC_DENSITY=0

The CEO may still issue many AskAgent calls per simulator turn (multi-step troubleshooting
in the prompt); that is model behavior, not mandate recursion. Cap tool rounds with
HASHIRU_MAX_TOOL_ROUNDS if needed (default 12 in manager.py).

Orchestration tracing (per task / per simulator step)
----------------------------------------------------
HASHIRU_GRADIO_URL sessions can log JSONL (see HASHIRU_ORCHESTRATION_TRACE_JSONL). The tau2
driver prepends a stripped ``[HASHIRU_TRACE_CTX]`` line on each Gradio /chat call with
``benchmark_name`` (e.g. tau2_airline), ``question_id`` (task id and trial when available),
and ``question_index`` (1-based step within that simulation). Vendored tau2-bench includes
trial in ``question_id``; pip-only tau2 installs still get task id and domain.

Troubleshooting
---------------
If import fails with: AttributeError: module 'aiohttp' has no attribute 'ConnectionTimeoutError'
(LiteLLM via tau2), upgrade aiohttp in the same venv you use to run this script, e.g.:
  pip install -U 'aiohttp>=3.10.0'
Ensure you activated the intended .venv (tracebacks may show a different path than this repo).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env so OPENAI_API_KEY (and other LiteLLM keys) are set for the user simulator
try:
    from dotenv import load_dotenv
    _bench_dir_for_dotenv = Path(__file__).resolve().parent
    load_dotenv(_bench_dir_for_dotenv / ".env")
    load_dotenv()  # also current working directory
except ImportError:
    pass

# Ensure bench dir is on path so hashiru_gradio_agent is importable
_bench_dir = Path(__file__).resolve().parent
if str(_bench_dir) not in sys.path:
    sys.path.insert(0, str(_bench_dir))


def _ensure_default_tau2_data_dir() -> None:
    """
    If TAU2_DATA_DIR is unset, point at vendored HASHIRU_Bench/bench/tau2-bench/data when present.

    Otherwise pip-installed tau2 often resolves DATA_DIR to a non-existent site-packages path.
    Must run before any ``import tau2`` (see run_benchmark order).
    """
    if (os.environ.get("TAU2_DATA_DIR") or "").strip():
        return
    vendored = (_bench_dir / "tau2-bench" / "data").resolve()
    if (vendored / "tau2" / "domains").is_dir():
        os.environ["TAU2_DATA_DIR"] = str(vendored)
        print(
            f"TAU2_DATA_DIR unset; using vendored tasks under {vendored}",
            file=sys.stderr,
        )


# Register HASHIRU agent with tau2 before any tau2 run imports that use the registry.
def _register_hashiru_agent():
    try:
        from tau2 import registry
    except ImportError as e:
        print("tau2-bench is not installed. Install with: pip install 'git+https://github.com/sierra-research/tau2-bench@main'", file=sys.stderr)
        raise SystemExit(1) from e

    from hashiru_gradio_agent import HashiruGradioAgent, TAU2_AVAILABLE
    if not TAU2_AVAILABLE or HashiruGradioAgent is None:
        raise SystemExit(1)
    try:
        registry.registry.register_agent(HashiruGradioAgent, "hashiru_agent")
    except ValueError:
        pass  # already registered (e.g. re-run in same process)


def _patch_tau2_run_task_for_hashiru_trace() -> None:
    """
    Ensure each simulation passes tau2_task_id and tau2_domain in llm_args_agent so
    hashiru_gradio_agent can prepend [HASHIRU_TRACE_CTX] for orchestration JSONL.

    Vendored tau2 (tau2-bench) also sets tau2_trial in run_tasks._run; pip installs get
    task id + domain from this patch only.
    """
    import tau2.run as tau2_run

    if getattr(tau2_run.run_task, "_hashiru_trace_merged", False):
        return
    _orig = tau2_run.run_task

    @functools.wraps(_orig)
    def _merged_llm_args(*args, **kwargs):
        merged = dict(kwargs.get("llm_args_agent") or {})
        task = kwargs.get("task")
        dom = kwargs.get("domain")
        if task is not None and getattr(task, "id", None) is not None:
            merged.setdefault("tau2_task_id", task.id)
        if dom:
            merged.setdefault("tau2_domain", dom)
        kwargs["llm_args_agent"] = merged
        return _orig(*args, **kwargs)

    _merged_llm_args._hashiru_trace_merged = True
    tau2_run.run_task = _merged_llm_args


def run_benchmark(
    domain: str,
    num_tasks: int | None = None,
    num_trials: int = 1,
    user_llm: str = "gpt-4.1",
    gradio_url: str = "http://127.0.0.1:7860",
    task_split: str = "base",
    task_ids: list[str] | None = None,
    save_to: str | None = None,
    max_steps: int = 100,
    max_concurrency: int = 1,
    seed: int | None = 300,
) -> dict:
    _ensure_default_tau2_data_dir()
    _register_hashiru_agent()
    _patch_tau2_run_task_for_hashiru_trace()

    from tau2.run import get_tasks, run_tasks
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.utils.utils import DATA_DIR

    # Check that tau2 data exists so we fail with a clear message instead of FileNotFoundError
    task_file = Path(DATA_DIR) / "tau2" / "domains" / domain / "tasks.json"
    if not task_file.is_file():
        print(
            "Tau2 data directory is missing or wrong. Tau2 expects task files at:\n"
            f"  {task_file}\n"
            "Set TAU2_DATA_DIR to the directory that contains a 'tau2' folder with domain data.\n"
            "Easiest: clone the repo and set (use your actual path):\n"
            "  export TAU2_DATA_DIR=/path/to/tau2-bench/data\n"
            "See the script docstring for 'Tau2 data directory setup'.",
            file=sys.stderr,
        )
        return {"error": "TAU2_DATA_DIR not set or data missing", "domain": domain, "expected": str(task_file)}

    task_set_name = domain
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=task_split,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )
    if not tasks:
        return {"error": "no tasks loaded", "domain": domain}

    llm_args_agent = {"gradio_url": gradio_url.rstrip("/")}

    results = run_tasks(
        domain=domain,
        tasks=tasks,
        agent="hashiru_agent",
        user="user_simulator",
        llm_agent="hashiru",
        llm_args_agent=llm_args_agent,
        llm_user=user_llm,
        llm_args_user={},
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=10,
        save_to=save_to,
        console_display=True,
        max_concurrency=max_concurrency,
        seed=seed,
    )

    metrics = compute_metrics(results)
    # Ensure JSON-serializable (metrics may be Pydantic)
    try:
        metrics_data = metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)
    except Exception:
        metrics_data = str(metrics)
    return {
        "domain": domain,
        "num_tasks": len(tasks),
        "num_trials": num_trials,
        "metrics": metrics_data,
        "simulations_count": len(results.simulations),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run τ²-bench with HASHIRU as the agent (Gradio at 127.0.0.1:7860).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="airline",
        choices=["airline", "retail", "telecom", "mock"],
        help="Tau2 domain to run (default: airline).",
    )
    parser.add_argument(
        "--num_tasks",
        "--num-tasks",
        type=int,
        default=None,
        help="Limit number of tasks (default: all in split).",
    )
    parser.add_argument(
        "--num_trials",
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1).",
    )
    parser.add_argument(
        "--user_llm",
        "--user-llm",
        type=str,
        default="gpt-4.1",
        help="LiteLLM model name for the user simulator (default: gpt-4.1).",
    )
    parser.add_argument(
        "--gradio_url",
        "--gradio-url",
        type=str,
        default=os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860"),
        help="HASHIRU Gradio URL (default: env HASHIRU_GRADIO_URL or http://127.0.0.1:7860).",
    )
    parser.add_argument(
        "--task_split",
        "--task-split",
        type=str,
        default="base",
        help="Task split (default: base for full benchmark set).",
    )
    parser.add_argument(
        "--task_ids",
        "--task-ids",
        type=str,
        default=None,
        nargs="+",
        help="Optional list of task IDs to run.",
    )
    parser.add_argument(
        "--save_to",
        "--save-to",
        type=str,
        default=None,
        help="Base name for tau2 simulation output (default: auto in results/).",
    )
    parser.add_argument(
        "--max_steps",
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per simulation (default: 100).",
    )
    parser.add_argument(
        "--max_concurrency",
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent simulations (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=300,
        help="Random seed (default: 300).",
    )
    args = parser.parse_args()

    # Default save_to under results/ (tau2 writes a .json file)
    bench_dir = Path(__file__).resolve().parent
    results_dir = bench_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.save_to is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_to = str(results_dir / f"tau2_{args.domain}_{timestamp}.json")

    summary = run_benchmark(
        domain=args.domain,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        user_llm=args.user_llm,
        gradio_url=args.gradio_url,
        task_split=args.task_split,
        task_ids=args.task_ids,
        save_to=args.save_to,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
    )

    out_path = results_dir / f"tau2_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.jsonl"
    with open(out_path, "w") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Summary written to {out_path}")
    if summary.get("error"):
        sys.exit(1)
    return summary


if __name__ == "__main__":
    main()
