#!/usr/bin/env python3
"""
MuSR (Multistep Soft Reasoning) benchmark for HASHIRU / HASSUM ablations.

MuSR provides long narrative + multi-step commonsense reasoning (murder mysteries,
object placement, team allocation). Strong fit for semantic uncertainty when agents
propose different reasoning chains.

Dataset: ``OpenEvals/MuSR`` (falls back to ``TAUR-Lab/MuSR``).

Usage:
  python3 benchmark_musr.py --domain murder_mysteries --num-questions 50 --offset 0 --out-dir musr_results
  python3 benchmark_musr.py --domain all --num-questions 30 --offset 0 --out-dir musr_results
"""
from __future__ import annotations

import argparse
import ast

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_MCQ
from benchmark_harness import (
    add_common_cli_args,
    benchmark_loop,
    extract_mcq_choice,
    min_question_seconds_from_env,
)

_MUSR_SOURCES = ("OpenEvals/MuSR", "TAUR-Lab/MuSR")
_DOMAINS = ("murder_mysteries", "object_placements", "team_allocation")


def _load_musr_split(domain: str):
    from datasets import load_dataset

    last_err = None
    for name in _MUSR_SOURCES:
        try:
            if name == "TAUR-Lab/MuSR":
                return load_dataset(name, split=domain), name
            return load_dataset(name, "default", split=domain), name
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load MuSR domain={domain}: {last_err}")


def _parse_choices(row: dict) -> list[str]:
    """Normalize MuSR choices/options (may be list or serialized string in parquet)."""
    raw = row.get("choices")
    if raw is None or (isinstance(raw, (list, tuple)) and not raw):
        raw = row.get("options")
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]


def _format_choices(choices: list[str]) -> tuple[str, dict[str, str]]:
    choice_map: dict[str, str] = {}
    lines: list[str] = []
    for i, text in enumerate(choices):
        if i >= 26:
            break
        letter = chr(ord("A") + i)
        choice_map[letter] = str(text)
        lines.append(f"{letter}. {text}")
    return "\n".join(lines), choice_map


def _gold_letter(row: dict, choice_map: dict[str, str]) -> str:
    if row.get("answer_choice"):
        ans = str(row["answer_choice"]).strip()
        for letter, text in choice_map.items():
            if ans == text or ans.upper() == letter:
                return letter
    idx = row.get("answer_index")
    if idx is not None:
        letters = list(choice_map.keys())
        if 0 <= int(idx) < len(letters):
            return letters[int(idx)]
    raise ValueError("Could not resolve gold choice for MuSR row")


def load_musr_items(domain: str, num_questions: int, offset: int) -> list[dict]:
    domains = _DOMAINS if domain == "all" else (domain,)
    all_items: list[dict] = []
    for dom in domains:
        ds, source = _load_musr_split(dom)
        for i in range(len(ds)):
            row = ds[i]
            choices = _parse_choices(row)
            if len(choices) < 2:
                continue
            choice_text, choice_map = _format_choices(choices)
            gold = _gold_letter(row, choice_map)
            all_items.append(
                {
                    "question_id": f"{dom}_{i}",
                    "dataset_index": i,
                    "question": row["question"],
                    "narrative": row.get("narrative", ""),
                    "choices_text": choice_text,
                    "choice_map": choice_map,
                    "gold_choice": gold,
                    "domain": dom,
                    "dataset_source": source,
                }
            )
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if offset >= len(all_items):
        raise ValueError(
            f"offset ({offset}) is out of range for {len(all_items)} MuSR items (domain={domain})."
        )
    return all_items[offset : offset + num_questions]


def build_prompt(item: dict) -> str:
    letters = ", ".join(sorted(item["choice_map"].keys()))
    return (
        CEO_FORCE_AGENTS_PREFIX_MCQ
        + "\n"
        + "You are solving a multi-step soft reasoning problem. Read the narrative carefully, "
        + "delegate sub-questions to agents, then select the best option.\n\n"
        + f"Narrative:\n{item['narrative']}\n\n"
        + f"Question: {item['question']}\n\n"
        + f"Options:\n{item['choices_text']}\n\n"
        + f'Reply with JSON only: {{"choice":"<LETTER>"}} where <LETTER> is one of {letters}.'
    )


def retry_worker(item: dict) -> str:
    return (
        "\n\n[Benchmark enforcement] Use AskAgent or AskMultipleAgents on the narrative, "
        "then output {\"choice\":\"<LETTER>\"}."
    )


def retry_format(item: dict) -> str:
    return '\n\n[Benchmark enforcement] Respond with JSON only: {"choice":"<LETTER>"}.'


def score_item(item: dict, parsed, mandate_violation: bool) -> dict:
    pred = (str(parsed).upper() if parsed else "")
    gold = item["gold_choice"]
    ok = bool(pred) and pred == gold and not mandate_violation
    return {
        "gold_choice": gold,
        "domain": item.get("domain"),
        "is_correct": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuSR benchmark against HASHIRU.")
    parser.add_argument(
        "--domain",
        default="murder_mysteries",
        choices=[*_DOMAINS, "all"],
        help="MuSR domain split (default: murder_mysteries).",
    )
    add_common_cli_args(parser)
    args = parser.parse_args()

    min_q = args.min_question_seconds
    if min_q is None:
        min_q = min_question_seconds_from_env(0.0)

    print(f"Loading MuSR (domain={args.domain})...")
    items = load_musr_items(args.domain, args.num_questions, args.offset)
    print(f"Loaded {len(items)} questions (domain={args.domain}, offset={args.offset})")

    benchmark_loop(
        benchmark_name="musr",
        benchmark_slug=f"musr_{args.domain}",
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
        extra_result_fields=lambda item: {"narrative_chars": len(item.get("narrative", ""))},
    )


if __name__ == "__main__":
    main()
