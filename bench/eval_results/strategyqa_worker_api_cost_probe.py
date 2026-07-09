#!/usr/bin/env python3
"""
Ballpark **worker** API cost: OpenAI (ChatGPT-class) vs **local** DeepSeek via Ollama,
on the same StrategyQA prompts — direct chat, no HASHIRU / Gemini orchestration.

OpenAI uses the cloud Chat Completions API (billed per token).
DeepSeek runs via Ollama (default ``http://127.0.0.1:11434``, model ``deepseek-r1``);
Ollama returns ``prompt_eval_count`` / ``eval_count`` so we can report tokens, and
local self-hosting is effectively $0/token. An optional self-hosted $/MTok rate is
exposed for compute-amortization estimates only.

Requires:
  - ``pip install openai datasets ollama`` (same stack the rest of HASHIRU uses)
  - Ollama running locally and the DeepSeek model pulled, e.g. ``ollama pull deepseek-r1``
  - ``OPENAI_API_KEY`` for OpenAI

Usage (from ``HASHIRU_Bench/bench``):
  python3 eval_results/strategyqa_worker_api_cost_probe.py
  python3 eval_results/strategyqa_worker_api_cost_probe.py --n 5 --offset 10
  python3 eval_results/strategyqa_worker_api_cost_probe.py --ollama-model deepseek-r1:7b

Pricing env (override after checking provider pages):

  OpenAI ($/1M tokens):
    HASHIRU_PROBE_OPENAI_INPUT_PER_MTOK  (default 2.50)
    HASHIRU_PROBE_OPENAI_OUTPUT_PER_MTOK (default 15.00)

  Self-hosted DeepSeek/Ollama (optional, for amortized compute estimates only):
    HASHIRU_PROBE_OLLAMA_INPUT_PER_MTOK  (default 0.00)
    HASHIRU_PROBE_OLLAMA_OUTPUT_PER_MTOK (default 0.00)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


def _f(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


SYSTEM_PROMPT = (
    "You answer StrategyQA-style questions. Reply with a single line of JSON only: "
    '{"answer":"yes"} or {"answer":"no"}. No other text.'
)


def load_strategyqa_rows(n: int, split: str, offset: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("ChilleD/StrategyQA", split=split)
    if offset >= len(ds):
        return []
    end = min(offset + n, len(ds))
    out = []
    for i in range(offset, end):
        row = ds[i]
        out.append(
            {
                "idx": i,
                "question": row["question"],
                "gold_yes": bool(row.get("answer", False)),
            }
        )
    return out


def _usage_from_openai_response(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    try:
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        return pt, ct
    except Exception:
        return 0, 0


def openai_complete(model: str, user_text: str) -> tuple[str, int, int]:
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    pt, ct = _usage_from_openai_response(resp)
    return text, pt, ct


def _ollama_field(resp: Any, name: str) -> Optional[int]:
    """ollama-python may return a pydantic-like object or a plain dict; handle both."""
    val = None
    if hasattr(resp, name):
        val = getattr(resp, name, None)
    if val is None and isinstance(resp, dict):
        val = resp.get(name)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _ollama_message_content(resp: Any) -> str:
    msg = getattr(resp, "message", None)
    if msg is not None:
        c = getattr(msg, "content", None)
        if c is None and isinstance(msg, dict):
            c = msg.get("content")
        if c is not None:
            return str(c)
    if isinstance(resp, dict):
        m = resp.get("message")
        if isinstance(m, dict):
            return str(m.get("content", ""))
        c = resp.get("response")
        if c is not None:
            return str(c)
    return ""


def ollama_complete(
    model: str, user_text: str, host: Optional[str] = None
) -> tuple[str, int, int, int]:
    """
    Returns (text, prompt_eval_count, eval_count, total_duration_ns).

    `prompt_eval_count` ≈ input tokens; `eval_count` ≈ output tokens (Ollama-reported).
    """
    import ollama

    client_kwargs: dict[str, Any] = {}
    if host:
        client_kwargs["host"] = host
    if client_kwargs:
        client = ollama.Client(**client_kwargs)
        chat_fn = client.chat
    else:
        chat_fn = ollama.chat

    resp = chat_fn(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        options={"temperature": 0.2},
    )
    text = _ollama_message_content(resp).strip()
    pt = _ollama_field(resp, "prompt_eval_count") or 0
    ct = _ollama_field(resp, "eval_count") or 0
    dur = _ollama_field(resp, "total_duration") or 0
    return text, pt, ct, dur


def usd_cost(prompt_tokens: int, completion_tokens: int, in_per_m: float, out_per_m: float) -> float:
    return (prompt_tokens / 1_000_000.0) * in_per_m + (completion_tokens / 1_000_000.0) * out_per_m


@dataclass
class Row:
    idx: int
    question: str
    openai_text: str = ""
    openai_pt: int = 0
    openai_ct: int = 0
    openai_usd: float = 0.0
    err_openai: Optional[str] = None
    ollama_text: str = ""
    ollama_pt: int = 0
    ollama_ct: int = 0
    ollama_usd: float = 0.0
    ollama_dur_ns: int = 0
    err_ollama: Optional[str] = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="Number of StrategyQA rows (default 5)")
    ap.add_argument("--split", default="test", help="ChilleD/StrategyQA split (default test)")
    ap.add_argument("--offset", type=int, default=0, help="Row offset in split")
    ap.add_argument(
        "--openai-model",
        default=os.environ.get("HASHIRU_OPENAI_CHATGPT_54_MODEL_ID", "").strip() or "gpt-5.4",
        help="OpenAI chat model id",
    )
    ap.add_argument(
        "--ollama-model",
        default=os.environ.get("HASHIRU_PROBE_OLLAMA_MODEL", "deepseek-r1"),
        help="Ollama model tag (must already be `ollama pull`-ed)",
    )
    ap.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "").strip() or None,
        help="Optional Ollama base URL (default uses ollama-python's default)",
    )
    args = ap.parse_args()

    openai_in = _f("HASHIRU_PROBE_OPENAI_INPUT_PER_MTOK", 2.50)
    openai_out = _f("HASHIRU_PROBE_OPENAI_OUTPUT_PER_MTOK", 15.00)
    ollama_in = _f("HASHIRU_PROBE_OLLAMA_INPUT_PER_MTOK", 0.00)
    ollama_out = _f("HASHIRU_PROBE_OLLAMA_OUTPUT_PER_MTOK", 0.00)

    rows_data = load_strategyqa_rows(args.n, args.split, args.offset)
    if not rows_data:
        print("No StrategyQA rows loaded (check split/offset).", file=sys.stderr)
        sys.exit(1)

    print("## StrategyQA worker probe — OpenAI cloud vs DeepSeek (Ollama, local)\n")
    print(f"- Split `{args.split}`, offset {args.offset}, n={len(rows_data)}")
    print(f"- OpenAI model: `{args.openai_model}`")
    print(f"- Ollama model: `{args.ollama_model}`  (host: `{args.ollama_host or 'default'}`)")
    print(
        "- **Cost assumptions (USD / 1M tokens)** — verify on provider sites:\n"
        f"  - OpenAI: input **{openai_in}**, output **{openai_out}** "
        "(env HASHIRU_PROBE_OPENAI_*)\n"
        f"  - Ollama (self-hosted): input **{ollama_in}**, output **{ollama_out}** "
        "(env HASHIRU_PROBE_OLLAMA_*; default 0 — local compute is not metered, "
        "set non-zero only if you want to amortize hardware/energy)\n"
    )

    out_rows: list[Row] = []
    sum_o_pt = sum_o_ct = sum_l_pt = sum_l_ct = 0
    sum_o_usd = sum_l_usd = 0.0
    sum_l_dur_ns = 0

    for item in rows_data:
        q = item["question"]
        user = (
            f'Answer with JSON only. Question: "{q}"\n'
            'Respond exactly: {"answer":"yes"} or {"answer":"no"}.'
        )
        r = Row(idx=item["idx"], question=q)

        try:
            text, pt, ct = openai_complete(args.openai_model, user)
            r.openai_text = text
            r.openai_pt, r.openai_ct = pt, ct
            r.openai_usd = usd_cost(pt, ct, openai_in, openai_out)
            sum_o_pt += pt
            sum_o_ct += ct
            sum_o_usd += r.openai_usd
        except Exception as e:
            r.err_openai = str(e)

        try:
            text, pt, ct, dur = ollama_complete(args.ollama_model, user, host=args.ollama_host)
            r.ollama_text = text
            r.ollama_pt, r.ollama_ct = pt, ct
            r.ollama_dur_ns = dur
            r.ollama_usd = usd_cost(pt, ct, ollama_in, ollama_out)
            sum_l_pt += pt
            sum_l_ct += ct
            sum_l_dur_ns += dur
            sum_l_usd += r.ollama_usd
        except Exception as e:
            r.err_ollama = str(e)

        out_rows.append(r)

    print("| # | dataset idx | OA prompt/out tok | OA $ | Ollama prompt/out tok | Ollama latency (s) | Ollama $ |")
    print("|---|-------------|--------------------|------|------------------------|--------------------|----------|")
    for i, r in enumerate(out_rows, 1):
        oa = f"{r.openai_pt}/{r.openai_ct}" if not r.err_openai else f"ERR"
        ll = f"{r.ollama_pt}/{r.ollama_ct}" if not r.err_ollama else f"ERR"
        latency = f"{r.ollama_dur_ns/1e9:.2f}" if r.ollama_dur_ns else ("—" if r.err_ollama else "0")
        print(
            f"| {i} | {r.idx} | {oa} | ${r.openai_usd:.6f} | "
            f"{ll} | {latency} | ${r.ollama_usd:.6f} |"
        )

    print("\n### Errors (if any)\n")
    any_err = False
    for r in out_rows:
        if r.err_openai:
            any_err = True
            print(f"- OpenAI idx={r.idx}: {r.err_openai}")
        if r.err_ollama:
            any_err = True
            print(f"- Ollama idx={r.idx}: {r.err_ollama}")
    if not any_err:
        print("- None")

    n_ok_o = sum(1 for r in out_rows if not r.err_openai)
    n_ok_l = sum(1 for r in out_rows if not r.err_ollama)
    print("\n### Totals (errors count as 0)\n")
    print("| Provider | Σ prompt | Σ output | n_ok | Σ wall (s) | Estimated $ |")
    print("|----------|----------|----------|------|------------|-------------|")
    print(f"| OpenAI ({args.openai_model}) | {sum_o_pt} | {sum_o_ct} | {n_ok_o} | — | **${sum_o_usd:.6f}** |")
    print(
        f"| Ollama ({args.ollama_model}) | {sum_l_pt} | {sum_l_ct} | {n_ok_l} | "
        f"{sum_l_dur_ns/1e9:.2f} | **${sum_l_usd:.6f}** |"
    )

    if sum_o_usd > 0 and sum_l_usd > 0:
        ratio = sum_o_usd / sum_l_usd
        print(f"\n**Ratio (OpenAI / Ollama) ≈ {ratio:.2f}×** at the supplied $/MTok rates "
              "(meaningful only if you set non-zero `HASHIRU_PROBE_OLLAMA_*`).")

    print("\n### Notes\n")
    print(
        "- Ollama token counts come from `prompt_eval_count` (input) and `eval_count` (output) "
        "in the chat response; these are model-tokenizer counts reported by the local server."
    )
    print(
        "- Self-hosted compute is not API-priced. To produce a defensible $/run for the local "
        "DeepSeek path, set HASHIRU_PROBE_OLLAMA_*_PER_MTOK based on a hardware/energy "
        "amortization model (e.g. GPU $/hour × throughput tokens/s)."
    )

    print("\n### Raw JSON\n")
    print(
        json.dumps(
            {
                "openai_model": args.openai_model,
                "ollama_model": args.ollama_model,
                "ollama_host": args.ollama_host,
                "pricing_usd_per_mtok": {
                    "openai_input": openai_in,
                    "openai_output": openai_out,
                    "ollama_input": ollama_in,
                    "ollama_output": ollama_out,
                },
                "totals": {
                    "openai": {"prompt": sum_o_pt, "completion": sum_o_ct, "usd": sum_o_usd, "n_ok": n_ok_o},
                    "ollama": {
                        "prompt": sum_l_pt,
                        "completion": sum_l_ct,
                        "usd": sum_l_usd,
                        "n_ok": n_ok_l,
                        "wall_seconds": sum_l_dur_ns / 1e9,
                    },
                },
                "rows": [
                    {
                        "dataset_idx": r.idx,
                        "openai": {"prompt": r.openai_pt, "completion": r.openai_ct, "usd": r.openai_usd, "error": r.err_openai},
                        "ollama": {
                            "prompt": r.ollama_pt,
                            "completion": r.ollama_ct,
                            "usd": r.ollama_usd,
                            "wall_ns": r.ollama_dur_ns,
                            "error": r.err_ollama,
                        },
                    }
                    for r in out_rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
