#!/usr/bin/env python3
"""
MuSiQue multi-hop QA benchmark for HASHIRU / HASSUM ablations.

MuSiQue enforces connected multi-hop reasoning (harder / less cheatable than HotpotQA).
Uses the answerable subset with provided paragraphs in the prompt (no retrieval).

Dataset: tries ``dgslibisey/MusiQue`` then ``MemoryAsModality/MuSiQue``.

Usage:
  python3 benchmark_musique.py --num-questions 50 --offset 0 --out-dir musique_results
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

_MUSIQUE_DATASETS = ("dgslibisey/MusiQue", "MemoryAsModality/MuSiQue")


def _load_musique_dataset(split: str):
    from datasets import load_dataset

    last_err = None
    for name in _MUSIQUE_DATASETS:
        try:
            return load_dataset(name, split=split), name
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load MuSiQue from {_MUSIQUE_DATASETS}: {last_err}")


def format_musique_paragraphs(paragraphs) -> str:
    blocks = []
    for p in paragraphs or []:
        if isinstance(p, dict):
            title = p.get("title", "")
            text = p.get("paragraph_text") or p.get("text") or ""
            blocks.append(f"Title: {title}\n{text}")
        else:
            blocks.append(str(p))
    return "\n\n".join(blocks)


def load_musique_items(split: str, num_questions: int, offset: int) -> list[dict]:
    ds, source = _load_musique_dataset(split)
    end = min(offset + num_questions, len(ds))
    if offset >= len(ds):
        raise ValueError(f"offset ({offset}) out of range for split size {len(ds)}")
    items = []
    for i in range(offset, end):
        row = ds[i]
        if row.get("answerable") is False:
            continue
        paragraphs = row.get("paragraphs")
        if paragraphs is None:
            paragraphs = row.get("context_paragraphs")
        items.append(
            {
                "question_id": row.get("id", str(i)),
                "dataset_index": i,
                "question": row["question"],
                "answer": row["answer"],
                "answerable": row.get("answerable", True),
                "context_text": format_musique_paragraphs(paragraphs),
                "dataset_source": source,
                "num_hops": len(row.get("question_decomposition") or []),
            }
        )
        if len(items) >= num_questions:
            break
    return items


def build_prompt(item: dict) -> str:
    hops = item.get("num_hops")
    hop_note = f"This question requires approximately {hops} reasoning hops. " if hops else ""
    return (
        CEO_FORCE_AGENTS_PREFIX_SHORT_ANSWER
        + "\n"
        + "You are answering a connected multi-hop question. "
        + hop_note
        + "Decompose the problem across agents using the context below.\n\n"
        + f"Context:\n{item['context_text']}\n\n"
        + f"Question: {item['question']}\n\n"
        + 'Reply with JSON only: {"answer":"<SHORT_ANSWER>"}.'
    )


def retry_worker(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Delegate to AskAgent or AskMultipleAgents before answering: "
        f'{item["question"]}'
    )


def retry_format(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Respond with JSON only: "
        f'{{"answer":"<SHORT_ANSWER>"}} for: {item["question"]}'
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
        "num_hops": item.get("num_hops"),
        "dataset_source": item.get("dataset_source"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuSiQue benchmark against HASHIRU.")
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation).")
    add_common_cli_args(parser)
    args = parser.parse_args()

    min_q = args.min_question_seconds
    if min_q is None:
        min_q = min_question_seconds_from_env(0.0)

    print("Loading MuSiQue...")
    items = load_musique_items(args.split, args.num_questions, args.offset)
    print(f"Loaded {len(items)} answerable questions (split={args.split}, offset={args.offset})")

    benchmark_loop(
        benchmark_name="musique",
        benchmark_slug="musique",
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
    )


if __name__ == "__main__":
    main()
