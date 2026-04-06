#!/usr/bin/env python3
"""One-off: compare jailbreak orig vs modified JSONL by sample_id."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
orig = HERE / "results" / "jailbreakbench_benchmark_20260317_120623_orig.jsonl"
mod = HERE / "results" / "jailbreakbench_benchmark_20260317_105009_modified.jsonl"


def load_map(p: Path) -> dict:
    m = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        m[o["sample_id"]] = o
    return m


def main() -> None:
    O, M = load_map(orig), load_map(mod)
    to_mod = [s for s in O if O[s]["is_correct"] is False and M[s]["is_correct"] is True]
    to_orig = [s for s in O if O[s]["is_correct"] is True and M[s]["is_correct"] is False]
    out = HERE / "_jailbreak_compare_out.txt"
    lines = [
        f"orig_false -> mod_true: {len(to_mod)}",
        f"orig_true -> mod_false: {len(to_orig)}",
    ]
    for sid in to_mod[:15]:
        lines.append(f"\n--- {sid} {O[sid]['category']}")
        lines.append("ORIG: " + (O[sid].get("agent_final_response") or "")[:500].replace("\n", " "))
        lines.append("MOD:  " + (M[sid].get("agent_final_response") or "")[:500].replace("\n", " "))
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
