#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACE = ROOT.parent / "results/semantic_metrics_logs/trace_20260409_174559_372695.jsonl"
OUT = ROOT / "semantic_metrics_trace_20260409_174559_playbyplay_20examples.txt"
RESULT_FILES = sorted(ROOT.glob("*_result.json"))


def letter(i: int) -> str:
    return chr(ord("A") + i)


def infer_is_correct(row: dict) -> bool | None:
    if "is_correct" in row and isinstance(row.get("is_correct"), bool):
        return row.get("is_correct")
    pred = row.get("pred")
    ans = row.get("answer")
    ans_idx = row.get("answer_index")
    if isinstance(pred, str):
        pred = pred.strip()
    if isinstance(ans, str):
        ans = ans.strip()
    if isinstance(pred, str) and isinstance(ans, str) and pred and ans:
        return pred == ans
    if isinstance(pred, str) and isinstance(ans_idx, int):
        return pred == letter(ans_idx)
    return None


def load_results() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fp in RESULT_FILES:
        try:
            data = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        subject = fp.name.replace("_result.json", "")
        for row in data:
            if not isinstance(row, dict) or "question_id" not in row:
                continue
            qid = str(row.get("question_id"))
            cur = out.get(qid)
            packed = {
                "subject": subject,
                "is_correct": infer_is_correct(row),
                "pred": row.get("pred"),
                "answer": row.get("answer"),
                "answer_index": row.get("answer_index"),
                "_file": fp.name,
            }
            if cur is None or (cur.get("is_correct") is None and packed.get("is_correct") is not None):
                out[qid] = packed
    return out


def load_trace_by_qid() -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    with TRACE.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = e.get("question_id")
            if qid is None:
                continue
            qid = str(qid)
            by.setdefault(qid, []).append(e)
    for qid in by:
        by[qid].sort(key=lambda x: float(x.get("ts") or 0.0))
    return by


def finishes(events: list[dict]) -> list[dict]:
    return [
        e
        for e in events
        if e.get("event") == "ceo_tool_finished" and e.get("tool") in ("AskAgent", "AskMultipleAgents")
    ]


def qtext(events: list[dict]) -> str:
    for e in events:
        t = e.get("question_text")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return ""


def score_item(fs: list[dict]) -> tuple[int, int, int, bool, int]:
    n = len(fs)
    rep = sum(1 for e in fs if e.get("worker_reprompted_after_semantic_check") is True)
    concerns = sum(
        1
        for e in fs
        if e.get("semantic_quality_concern") is True
        or "crossed thresholds" in str(e.get("message") or "").lower()
    )
    has_multi = any(e.get("tool") == "AskMultipleAgents" for e in fs)
    errs = sum(1 for e in fs if str(e.get("status")) == "error")
    rank = n + rep + concerns + (3 if has_multi else 0) + (2 * errs)
    return rank, n, rep, has_multi, concerns


def fmt_metrics(e: dict) -> str:
    keys = [
        "semantic_entropy",
        "semantic_density",
        "semantic_entropy_threshold",
        "semantic_density_threshold",
        "semantic_quality_concern",
        "worker_reprompted_after_semantic_check",
    ]
    parts = [f"{k}={e.get(k)!r}" for k in keys if k in e]
    return "; ".join(parts) if parts else "(no top-level semantic fields)"


def emit_round(buf: list[str], i: int, e: dict) -> None:
    tool = e.get("tool")
    buf.append(f"\n--- Worker round {i + 1} (ceo_tool_finished, tool={tool}) ---\n")
    buf.append(f"Tool status/message: {e.get('status')!r} / {e.get('message')!r}\n")
    buf.append(f"Metrics: {fmt_metrics(e)}\n")
    if tool == "AskAgent":
        buf.append(f"Agent: {e.get('agent_name')!r}\n")
        buf.append("\nPROMPT (exact):\n")
        buf.append(str(e.get("worker_prompt") or (e.get("args") or {}).get("prompt") or ""))
        buf.append("\n\nRESPONSE (exact):\n")
        buf.append(str(e.get("worker_response") or ""))
        buf.append("\n")
    elif tool == "AskMultipleAgents":
        uq = e.get("user_question") or (e.get("args") or {}).get("user_question")
        if uq:
            buf.append(f"\nUser question passed to tool (exact):\n{uq}\n")
        pa = e.get("per_agent_outputs")
        if isinstance(pa, list):
            for j, row in enumerate(pa):
                if not isinstance(row, dict):
                    continue
                buf.append(
                    f"\n--- Sub-agent {j + 1}: {row.get('agent_name')!r} (base_model={row.get('base_model')!r}) ---\n"
                )
                buf.append("\nPROMPT (exact):\n")
                buf.append(str(row.get("prompt") or ""))
                buf.append("\n\nRESPONSE (exact):\n")
                buf.append(str(row.get("response") or ""))
                buf.append("\n")
                if "semantic_quality_concern" in row:
                    buf.append(
                        f"(per-agent: semantic_quality_concern={row.get('semantic_quality_concern')!r}, "
                        f"entropy={row.get('semantic_entropy')!r}, density={row.get('semantic_density')!r})\n"
                    )


def play(qid: str, idx: int, events: list[dict], res: dict | None) -> str:
    fs = finishes(events)
    rank, n, rep, has_multi, concerns = score_item(fs)
    label = f"M{idx:02d} — qid {qid}"
    if has_multi:
        label += " | includes AskMultipleAgents"
    if rep > 0:
        label += f" | reprompts={rep}"
    bench = "no result row found in *_result.json"
    if res:
        bench = (
            f"subject={res.get('subject')} | is_correct={res.get('is_correct')!r} | "
            f"pred={res.get('pred')!r} | answer={res.get('answer')!r} | source={res.get('_file')}"
        )
    lines = []
    lines.append("=" * 88 + "\n")
    lines.append(f"QUESTION ID: {qid}\n")
    lines.append(f"Section label: {label}\n")
    lines.append(f"Worker tool rounds (AskAgent / AskMultipleAgents): {n}\n")
    lines.append(f"CEO reprompts after first worker round: {max(0, n-1)}\n")
    lines.append(f"Trace file: {TRACE.name}\n")
    lines.append(f"Benchmark / scoring note: {bench}\n")
    lines.append("\n--- ORIGINAL BENCHMARK QUESTION (question_text) ---\n")
    lines.append(qtext(events) + "\n\n")
    lines.append("--- HIGH-LEVEL SUMMARY ---\n")
    lines.append(
        f"Interestingness rank={rank}; concerns={concerns}; has_multi={has_multi}; reprompt_true_flags={rep}.\n"
    )
    if res and isinstance(res.get("is_correct"), bool):
        lines.append("Outcome: correct.\n" if res.get("is_correct") else "Outcome: incorrect.\n")
    lines.append("\n--- FULL PLAY-BY-PLAY (exact prompts and responses) ---\n")
    for i, e in enumerate(fs):
        emit_round(lines, i, e)
    return "".join(lines)


def choose_20(items: list[dict]) -> list[dict]:
    helpful = [
        x
        for x in items
        if x["res"]
        and x["res"].get("is_correct") is True
        and (x["rep"] > 0 or x["has_multi"] or x["concerns"] > 0)
    ]
    harmful = [x for x in items if x["res"] and x["res"].get("is_correct") is False]
    neutral = [
        x
        for x in items
        if x["res"] and x["res"].get("is_correct") is True and x["rep"] == 0 and not x["has_multi"]
    ]
    missing = [x for x in items if x["res"] is None]

    helpful.sort(key=lambda x: x["rank"], reverse=True)
    harmful.sort(key=lambda x: x["rank"], reverse=True)
    neutral.sort(key=lambda x: x["rank"], reverse=True)
    missing.sort(key=lambda x: x["rank"], reverse=True)

    picked: list[dict] = []
    used = set()

    def take(pool: list[dict], k: int) -> None:
        nonlocal picked
        for x in pool:
            if len(picked) >= 20 or k <= 0:
                break
            if x["qid"] in used:
                continue
            picked.append(x)
            used.add(x["qid"])
            k -= 1

    take(helpful, 12)
    take(harmful, 6)
    take(neutral, 2)
    if len(picked) < 20:
        take(helpful, 20)
    if len(picked) < 20:
        take(harmful, 20)
    if len(picked) < 20:
        take(neutral, 20)
    if len(picked) < 20:
        take(missing, 20)
    return picked[:20]


def main() -> None:
    results = load_results()
    by_qid = load_trace_by_qid()

    items = []
    for qid, evs in by_qid.items():
        fs = finishes(evs)
        if not fs:
            continue
        rank, n, rep, has_multi, concerns = score_item(fs)
        items.append(
            {
                "qid": qid,
                "events": evs,
                "rank": rank,
                "n": n,
                "rep": rep,
                "has_multi": has_multi,
                "concerns": concerns,
                "res": results.get(qid),
            }
        )

    chosen = choose_20(items)

    hdr = []
    hdr.append("Semantic metrics trace play-by-play (20 interesting MMLU examples)\n")
    hdr.append("=" * 88 + "\n")
    hdr.append(f"Trace: {TRACE.name}\n")
    hdr.append("Result joins: all *_result.json files under eval_results (subject inferred from filename).\n")
    hdr.append("Selection intent: prioritize cases where semantic metrics appear to guide retries/delegation,\n")
    hdr.append("while keeping several counterexamples where impact is limited or appears harmful.\n\n")
    hdr.append("--- MASTER SUMMARY (selected 20 question_ids) ---\n")
    for i, x in enumerate(chosen, 1):
        qid = x["qid"]
        res = x["res"] or {}
        txt = qtext(x["events"])[:90].replace("\n", " ")
        hdr.append(
            f"M{i:02d} qid={qid}: rounds={x['n']}, reprompt_true={x['rep']}, "
            f"multi={x['has_multi']}, concerns={x['concerns']}, "
            f"subject={res.get('subject')}, is_correct={res.get('is_correct')} | {txt}...\n"
        )
    hdr.append("\n")

    parts = ["".join(hdr)]
    for i, x in enumerate(chosen, 1):
        parts.append(play(x["qid"], i, x["events"], x["res"]))
        parts.append("\n")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
