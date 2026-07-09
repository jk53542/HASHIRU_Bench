#!/usr/bin/env python3
"""
BIG-Bench Hard (BBH) benchmark for HASHIRU / HASSUM ablations.

BBH is a collection of 27 challenging reasoning tasks. This script runs a curated
``hassum`` subset where multi-step reasoning and answer uncertainty are most relevant,
or individual tasks via ``--tasks``.

Dataset: ``Joschka/big_bench_hard`` (per-task configs).

Usage:
  python3 benchmark_bbh.py --task-subset hassum --num-questions 20 --offset 0 --out-dir bbh_results
  python3 benchmark_bbh.py --tasks logical_deduction_three_objects,multistep_arithmetic_two --num-questions 50 --offset 0
"""
from __future__ import annotations

import argparse

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_MCQ, CEO_FORCE_AGENTS_PREFIX_SHORT_ANSWER
from benchmark_harness import (
    add_common_cli_args,
    benchmark_loop,
    extract_mcq_choice,
    extract_short_answer,
    min_question_seconds_from_env,
)

# Config names on Joschka/big_bench_hard (see dataset card for full list).
HASSUM_TASK_SUBSET = (
    "logical_deduction_three_objects",
    "tracking_shuffled_objects_three_objects",
    "multistep_arithmetic_two",
    "causal_judgement",
    "date_understanding",
    "navigate",
    "web_of_lies",
    "temporal_sequences",
    "object_counting",
    "disambiguation_qa",
)

ALL_BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

# Backward-compatible aliases for older task names used in docs/scripts.
_BBH_TASK_ALIASES = {
    "logical_deduction": "logical_deduction_three_objects",
    "multi_step_arithmetic": "multistep_arithmetic_two",
    "causal_judgment": "causal_judgement",
    "formal_fallacies_syllogisms_negation": "formal_fallacies",
}


def _resolve_bbh_task(task: str) -> str:
    t = task.strip()
    return _BBH_TASK_ALIASES.get(t, t)


def _parse_target(target: str) -> tuple[str, str]:
    """Return (mode, value) where mode is 'mcq' or 'short'."""
    t = (target or "").strip()
    if not t:
        return "short", ""
    if len(t) == 1 and t.upper() in "ABCDE":
        return "mcq", t.upper()
    if t.startswith("(") and ")" in t:
        inner = t[1 : t.index(")")]
        if len(inner) == 1 and inner.upper() in "ABCDE":
            return "mcq", inner.upper()
    return "short", t


def load_bbh_items(tasks: list[str], num_questions: int, offset: int) -> list[dict]:
    from datasets import load_dataset

    if offset < 0:
        raise ValueError("offset must be >= 0")

    all_items: list[dict] = []
    for task in tasks:
        resolved = _resolve_bbh_task(task)
        # Each BBH config is its own split (there is no shared "test" split).
        ds = load_dataset("Joschka/big_bench_hard", resolved, split=resolved)
        for i in range(len(ds)):
            row = ds[i]
            mode, gold = _parse_target(str(row.get("target", "")))
            all_items.append(
                {
                    "question_id": f"{resolved}_{i}",
                    "dataset_index": i,
                    "question": str(row.get("input", "")),
                    "gold_answer": gold,
                    "answer_mode": mode,
                    "task": resolved,
                }
            )
    if offset >= len(all_items):
        raise ValueError(
            f"offset ({offset}) is out of range for {len(all_items)} BBH items "
            f"across {len(tasks)} task(s)."
        )
    return all_items[offset : offset + num_questions]


def build_prompt(item: dict) -> str:
    mode = item["answer_mode"]
    if mode == "mcq":
        return (
            CEO_FORCE_AGENTS_PREFIX_MCQ
            + "\n"
            + f"BIG-Bench Hard task: {item['task']}\n\n"
            + f"{item['question']}\n\n"
            + 'Reply with JSON only: {"choice":"<LETTER>"}.'
        )
    return (
        CEO_FORCE_AGENTS_PREFIX_SHORT_ANSWER
        + "\n"
        + f"BIG-Bench Hard task: {item['task']}\n\n"
        + f"{item['question']}\n\n"
        + 'Reply with JSON only: {"answer":"<ANSWER>"}.'
    )


def parse_response(history):
    choice = extract_mcq_choice(history)
    if choice:
        return choice
    return extract_short_answer(history)


def retry_worker(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Use AskAgent or AskMultipleAgents, then output the "
        "required JSON answer for this BBH item."
    )


def retry_format(item: dict) -> str:
    if item["answer_mode"] == "mcq":
        return '\n\n[Benchmark enforcement] Respond with {"choice":"<LETTER>"} only.'
    return '\n\n[Benchmark enforcement] Respond with {"answer":"<ANSWER>"} only.'


def score_item(item: dict, parsed, mandate_violation: bool) -> dict:
    gold = item["gold_answer"]
    pred = str(parsed or "").strip()
    if item["answer_mode"] == "mcq":
        ok = bool(pred) and pred.upper() == gold.upper() and not mandate_violation
    else:
        ok = bool(pred) and pred.lower() == gold.lower() and not mandate_violation
    return {
        "gold_answer": gold,
        "answer_mode": item["answer_mode"],
        "task": item["task"],
        "is_correct": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIG-Bench Hard benchmark against HASHIRU.")
    parser.add_argument(
        "--task-subset",
        default="hassum",
        choices=["hassum", "all"],
        help="Predefined task list (default: hassum = reasoning-heavy subset).",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated BBH task names (overrides --task-subset).",
    )
    add_common_cli_args(parser)
    args = parser.parse_args()

    if args.tasks.strip():
        tasks = [_resolve_bbh_task(t) for t in args.tasks.split(",") if t.strip()]
    elif args.task_subset == "all":
        tasks = list(ALL_BBH_TASKS)
    else:
        tasks = list(HASSUM_TASK_SUBSET)

    min_q = args.min_question_seconds
    if min_q is None:
        min_q = min_question_seconds_from_env(0.0)

    print(f"Loading BBH tasks: {', '.join(tasks)}")
    items = load_bbh_items(tasks, args.num_questions, args.offset)
    print(f"Loaded {len(items)} items (offset={args.offset}, tasks={len(tasks)})")

    benchmark_loop(
        benchmark_name="bbh",
        benchmark_slug="bbh",
        items=items,
        out_dir=args.out_dir,
        build_prompt=build_prompt,
        parse_response=parse_response,
        score_item=score_item,
        require_worker_tools=not args.allow_ceo_only,
        max_retries=max(1, args.max_retries),
        min_question_seconds=min_q,
        retry_suffix_worker=retry_worker,
        retry_suffix_format=retry_format,
    )


if __name__ == "__main__":
    main()
