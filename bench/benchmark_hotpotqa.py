#!/usr/bin/env python3
"""
HotpotQA (distractor setting) benchmark for HASHIRU / HASSUM ablations.

Dataset: ``hotpot_qa`` config ``distractor`` — multi-hop questions with provided
paragraphs (no retrieval required). Scored with exact match + token F1.

Usage (from ``HASHIRU_Bench/bench`` with HASHIRU running on :7860):
  python3 benchmark_hotpotqa.py --num-questions 50 --offset 0 --out-dir hotpotqa_results
"""
from __future__ import annotations

import argparse

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_SHORT_ANSWER
from benchmark_harness import (
    add_common_cli_args,
    benchmark_loop,
    exact_match_score,
    extract_short_answer,
    f1_score,
    min_question_seconds_from_env,
)


def format_hotpot_context(context: dict) -> str:
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    blocks = []
    for title, sents in zip(titles, sentences):
        para = " ".join(sents)
        blocks.append(f"Title: {title}\n{para}")
    return "\n\n".join(blocks)


def load_hotpotqa_items(split: str, num_questions: int, offset: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    end = min(offset + num_questions, len(ds))
    if offset >= len(ds):
        raise ValueError(f"offset ({offset}) out of range for split size {len(ds)}")
    items = []
    for i in range(offset, end):
        row = ds[i]
        items.append(
            {
                "question_id": row.get("id", str(i)),
                "dataset_index": i,
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type"),
                "level": row.get("level"),
                "context_text": format_hotpot_context(row.get("context") or {}),
            }
        )
    return items


def build_prompt(item: dict) -> str:
    return (
        CEO_FORCE_AGENTS_PREFIX_SHORT_ANSWER
        + "\n"
        + "You are answering a multi-hop question. Use the provided context paragraphs; "
        + "reason step by step via agents, then give a short final answer.\n\n"
        + f"Context:\n{item['context_text']}\n\n"
        + f"Question: {item['question']}\n\n"
        + 'Reply with JSON only: {"answer":"<SHORT_ANSWER>"}. '
        + "Use a concise span (name, date, number, or short phrase)."
    )


def retry_worker(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Call GetAgents if needed, then AskAgent or "
        "AskMultipleAgents, read worker output, and only then output "
        f'{{"answer":"..."}} for: {item["question"]}'
    )


def retry_format(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Previous response was not valid JSON with an answer field. "
        f'Question: {item["question"]} — respond with {{"answer":"<SHORT_ANSWER>"}} only.'
    )


def score_item(item: dict, parsed, mandate_violation: bool) -> dict:
    pred = str(parsed or "")
    gold = item["answer"]
    em = bool(pred) and exact_match_score(pred, gold) and not mandate_violation
    f1 = f1_score(pred, gold) if pred else 0.0
    return {
        "gold_answer": gold,
        "exact_match": em,
        "f1": f1,
        "is_correct": em,
        "type": item.get("type"),
        "level": item.get("level"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA (distractor) benchmark against HASHIRU.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    add_common_cli_args(parser)
    args = parser.parse_args()

    min_q = args.min_question_seconds
    if min_q is None:
        min_q = min_question_seconds_from_env(0.0)

    print("Loading HotpotQA (distractor)...")
    items = load_hotpotqa_items(args.split, args.num_questions, args.offset)
    print(f"Loaded {len(items)} questions (split={args.split}, offset={args.offset})")

    benchmark_loop(
        benchmark_name="hotpotqa",
        benchmark_slug="hotpotqa",
        items=items,
        out_dir=args.out_dir,
        build_prompt=build_prompt,
        parse_response=extract_short_answer,
        score_item=score_item,
        require_worker_tools=not args.allow_ceo_only,
        max_retries=max(1, args.max_retries),
        min_question_seconds=min_q,
        retry_suffix_worker=retry_worker,
        retry_suffix_format=retry_format,
        extra_result_fields=lambda item: {"context_chars": len(item.get("context_text", ""))},
    )


if __name__ == "__main__":
    main()
