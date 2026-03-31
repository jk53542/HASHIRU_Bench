#!/usr/bin/env python3
"""
THUDM AgentBench (classic controller + assigner) with HASHIRU as the agent.

This mirrors the tau2/BFCL pattern: adapter code lives in this repo; you clone
AgentBench separately and point --agentbench-root at it.

Prerequisites (from AgentBench docs):
  - Clone https://github.com/THUDM/AgentBench
  - Install its Python deps (`pip install -r requirements.txt`)
  - Build/pull Docker images and start task workers + controller, e.g.:
      python -m src.start_task -a
    (or the lite preset). The assigner talks to `http://localhost:5000/api` by default.

HASHIRU:
  - Run HASHIRU Gradio (default http://127.0.0.1:7860)
  - This script configures modes with budgets OFF and tool invocation ON, and prepends
    the CEO "You MUST use agents..." instructions on every inference call.

Usage (from anywhere):
  python benchmarking_agentbench.py --agentbench-root /path/to/AgentBench

Optional:
  python benchmarking_agentbench.py --agentbench-root ... --tasks dbbench-std,os-std --agent-concurrency 1
  python benchmarking_agentbench.py --agentbench-root ... --write-config-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _yaml_list_indent(items: list[str], indent: str) -> str:
    return "\n".join(f'{indent}- {x}' for x in items)


def write_agentbench_hashiru_configs(
    agentbench_root: Path,
    gradio_url: str,
    tasks: list[str],
    agent_concurrency: int,
    task_concurrency: int,
    output_glob: str,
) -> Path:
    """
    Writes:
      configs/agents/hashiru_gradio.yaml
      configs/assignments/hashiru_gradio_assignment.yaml
    Returns path to the assignment yaml (relative to agentbench_root).
    """
    agents_dir = agentbench_root / "configs" / "agents"
    assign_dir = agentbench_root / "configs" / "assignments"

    agent_yaml = agents_dir / "hashiru_gradio.yaml"
    agent_body = (
        "hashiru-gradio:\n"
        "  module: hashiru_agentbench_client.HashiruGradioAgentBenchClient\n"
        "  parameters:\n"
        f'    gradio_url: "{gradio_url}"\n'
    )
    _write_text(agent_yaml, agent_body)

    task_conc_lines = "\n".join(
        f"    {t}: {task_concurrency}" for t in tasks
    )
    task_assign_lines = _yaml_list_indent(tasks, "      ")

    assignment_path = assign_dir / "hashiru_gradio_assignment.yaml"
    assignment_body = (
        "import: definition.yaml\n"
        "\n"
        "definition:\n"
        "  agent:\n"
        "    import: ../agents/hashiru_gradio.yaml\n"
        "\n"
        "concurrency:\n"
        "  task:\n"
        f"{task_conc_lines}\n"
        "  agent:\n"
        f"    hashiru-gradio: {agent_concurrency}\n"
        "\n"
        "assignments:\n"
        "  - agent:\n"
        "      - hashiru-gradio\n"
        "    task:\n"
        f"{task_assign_lines}\n"
        "\n"
        f'output: "{output_glob}"\n'
    )
    _write_text(assignment_path, assignment_body)
    return assignment_path.relative_to(agentbench_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run THUDM AgentBench assigner with HASHIRU (Gradio) as the agent client.",
    )
    parser.add_argument(
        "--agentbench-root",
        type=str,
        default=os.environ.get("AGENTBENCH_ROOT", ""),
        help="Path to a clone of https://github.com/THUDM/AgentBench (or set AGENTBENCH_ROOT).",
    )
    parser.add_argument(
        "--gradio-url",
        type=str,
        default=os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860"),
        help="HASHIRU Gradio base URL (no trailing slash required).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="dbbench-std,os-std",
        help="Comma-separated AgentBench task names (default: dbbench-std,os-std).",
    )
    parser.add_argument(
        "--agent-concurrency",
        type=int,
        default=1,
        help="Assigner concurrency for the HASHIRU agent (default: 1).",
    )
    parser.add_argument(
        "--task-concurrency",
        type=int,
        default=1,
        help="Declared per-task concurrency in the assignment (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default='outputs/hashiru_{TIMESTAMP}',
        help='Assigner output path template (default: outputs/hashiru_{TIMESTAMP}).',
    )
    parser.add_argument(
        "--write-config-only",
        action="store_true",
        help="Only write YAML configs under AgentBench; do not run the assigner.",
    )
    parser.add_argument(
        "--auto-retry",
        action="store_true",
        help="Forward --auto-retry to AgentBench assigner (retry failed samples).",
    )
    args = parser.parse_args()

    if not args.agentbench_root:
        print(
            "Error: pass --agentbench-root /path/to/AgentBench or set AGENTBENCH_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    agentbench_root = Path(args.agentbench_root).expanduser().resolve()
    if not (agentbench_root / "src" / "assigner.py").is_file():
        print(
            f"Error: {agentbench_root} does not look like AgentBench (missing src/assigner.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    bench_dir = Path(__file__).resolve().parent
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        print("Error: --tasks resulted in an empty list.", file=sys.stderr)
        sys.exit(1)

    rel_assign = write_agentbench_hashiru_configs(
        agentbench_root=agentbench_root,
        gradio_url=args.gradio_url.rstrip("/"),
        tasks=tasks,
        agent_concurrency=args.agent_concurrency,
        task_concurrency=args.task_concurrency,
        output_glob=args.output,
    )
    print(f"Wrote AgentBench configs under {agentbench_root}")
    print(f"  Assignment: {rel_assign.as_posix()}")

    if args.write_config_only:
        print("write-config-only: skipping assigner.")
        return

    env = os.environ.copy()
    extra = f"{str(bench_dir)}{os.pathsep}{str(agentbench_root)}"
    env["PYTHONPATH"] = (
        extra if "PYTHONPATH" not in env or not env["PYTHONPATH"] else f"{extra}{os.pathsep}{env['PYTHONPATH']}"
    )

    cmd = [
        sys.executable,
        "-m",
        "src.assigner",
        "--config",
        str(rel_assign.as_posix()),
    ]
    if args.auto_retry:
        cmd.append("--auto-retry")

    print("Starting AgentBench assigner (official evaluation loop)…")
    print(f"  cwd={agentbench_root}")
    print(f"  cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(agentbench_root), env=env, check=True)


if __name__ == "__main__":
    main()
