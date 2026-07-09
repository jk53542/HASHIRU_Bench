#!/usr/bin/env python3
"""
GPQA multiple-choice benchmark for HASHIRU / HASSUM ablations.

GPQA (graduate-level science) is very hard; models often disagree or hedge — a good
stress test for semantic entropy / density when routing among worker agents.

Dataset: ``Idavidrein/gpqa`` (default config ``gpqa_diamond``, 198 questions).
Requires Hugging Face login — visit the dataset page and accept the terms, then
``huggingface-cli login``.

Usage:
  python3 benchmark_gpqa.py --config gpqa_diamond --num-questions 50 --offset 0 --out-dir gpqa_results
"""
from __future__ import annotations

import argparse
import random

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_MCQ
from benchmark_harness import (
    add_common_cli_args,
    benchmark_loop,
    extract_mcq_choice,
    min_question_seconds_from_env,
)

_CHOICE_LABELS = ("A", "B", "C", "D")
_GPQA_ANSWER_COLS = (
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)


def _gpqa_correct_and_distractors(row: dict) -> tuple[str, list[str]]:
    """Return (correct_text, [distractor1, distractor2, distractor3]) from GPQA CSV columns."""
    correct = str(row.get("Correct Answer") or row.get("correct answer") or "").strip()
    distractors = [
        str(row.get(f"Incorrect Answer {i}") or row.get(f"incorrect answer {i}") or "").strip()
        for i in (1, 2, 3)
    ]
    distractors = [d for d in distractors if d]
    if not correct or len(distractors) != 3:
        raise ValueError("GPQA row missing Correct Answer or three incorrect options")
    return correct, distractors


def _gpqa_choices_and_gold(row: dict, *, shuffle_seed: int) -> tuple[list[str], str]:
    """
    Build four shuffled MCQ options and the gold letter.

    GPQA stores the key in ``Correct Answer`` (not answer_index). Standard eval practice
    shuffles the four options so the correct answer is not always in the same slot.
    """
    correct, distractors = _gpqa_correct_and_distractors(row)
    choices = [correct, *distractors]
    rng = random.Random(shuffle_seed)
    rng.shuffle(choices)
    gold_idx = choices.index(correct)
    if gold_idx >= len(_CHOICE_LABELS):
        raise ValueError(f"Unexpected choice count after shuffle: {len(choices)}")
    return choices, _CHOICE_LABELS[gold_idx]


def load_gpqa_items(config: str, num_questions: int, offset: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    try:
        ds = load_dataset("Idavidrein/gpqa", config, split="train", token=True)
    except DatasetNotFoundError as e:
        raise RuntimeError(
            "GPQA (Idavidrein/gpqa) is a gated Hugging Face dataset. "
            "1) Visit https://huggingface.co/datasets/Idavidrein/gpqa and accept the terms. "
            "2) Run: huggingface-cli login  (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN). "
            "3) Re-run this script."
        ) from e
    indices = list(range(len(ds)))
    # GPQA has no official val split; use deterministic shuffle for held-out-style slices.
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = indices[offset : offset + num_questions]
    items = []
    for i in selected:
        row = ds[i]
        try:
            choices, gold = _gpqa_choices_and_gold(row, shuffle_seed=seed + i)
        except ValueError:
            continue
        letter_map = {lab: choices[j] for j, lab in enumerate(_CHOICE_LABELS)}
        question = str(row.get("Question") or row.get("question") or "").strip()
        if not question:
            continue
        items.append(
            {
                "question_id": str(row.get("id") or f"{config}_{i}"),
                "dataset_index": i,
                "question": question,
                "choices": choices,
                "choice_map": letter_map,
                "gold_choice": gold,
                "domain": row.get("Subdomain") or row.get("subdomain") or row.get("field"),
            }
        )
    return items


def _format_choices(choice_map: dict[str, str]) -> str:
    lines = []
    for letter in sorted(choice_map.keys()):
        lines.append(f"{letter}. {choice_map[letter]}")
    return "\n".join(lines)


def build_prompt(item: dict) -> str:
    return (
        CEO_FORCE_AGENTS_PREFIX_MCQ
        + "\n"
        + "You are solving a graduate-level multiple-choice science question. "
        + "Consult agents with relevant expertise, then pick one option.\n\n"
        + f"Question: {item['question']}\n\n"
        + f"Options:\n{_format_choices(item['choice_map'])}\n\n"
        + 'Reply with JSON only: {"choice":"<LETTER>"} where <LETTER> is A, B, C, or D.'
    )


def retry_worker(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Use AskAgent or AskMultipleAgents, then output "
        f'{{"choice":"<LETTER>"}} for: {item["question"][:200]}'
    )


def retry_format(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Respond with JSON only: "
        '{"choice":"<LETTER>"} (A–D).'
    )


def score_item(item: dict, parsed, mandate_violation: bool) -> dict:
    pred = (str(parsed).upper() if parsed else "")
    gold = item["gold_choice"]
    ok = bool(pred) and pred == gold and not mandate_violation
    return {
        "gold_choice": gold,
        "choice_map": item["choice_map"],
        "domain": item.get("domain"),
        "is_correct": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPQA benchmark against HASHIRU.")
    parser.add_argument(
        "--config",
        default="gpqa_diamond",
        choices=["gpqa_main", "gpqa_diamond", "gpqa_extended"],
        help="GPQA subset (default: gpqa_diamond).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for slicing train split.")
    add_common_cli_args(parser)
    args = parser.parse_args()

    min_q = args.min_question_seconds
    if min_q is None:
        min_q = min_question_seconds_from_env(0.0)

    print(f"Loading GPQA ({args.config})...")
    items = load_gpqa_items(args.config, args.num_questions, args.offset, args.seed)
    print(
        f"Loaded {len(items)} questions "
        f"(config={args.config}, offset={args.offset}, seed={args.seed})"
    )

    benchmark_loop(
        benchmark_name="gpqa",
        benchmark_slug=f"gpqa_{args.config}",
        items=items,
        out_dir=args.out_dir,
        build_prompt=build_prompt,
        parse_response=extract_mcq_choice,
        score_item=score_item,
        require_worker_tools=not args.allow_ceo_only,
        max_retries=max(1, args.max_retries),
        min_question_seconds=min_q,
        retry_suffix_worker=retry_worker,
        retry_suffix_format=retry_format,
    )


if __name__ == "__main__":
    main()
