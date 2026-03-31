#!/usr/bin/env python3
"""
Gorilla BFCL (Berkeley Function Calling Leaderboard) runner using HASHIRU.

BFCL **V4** (see `bfcl_eval.constants.category_mapping`) includes more than single-turn AST
function-calling:

- **Single-turn (non-live / live)**: e.g. `simple_python`, `multiple`, `parallel`, …
- **Multi-turn** (`MULTI_TURN_CATEGORY`): conversational / multi-step FC — `multi_turn_base`,
  `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`.
- **Agentic** (`AGENTIC_CATEGORY`): memory + web-search style scenarios — `memory_kv`,
  `memory_vector`, `memory_rec_sum`, `web_search_base`, `web_search_no_snippet`.
- **Format sensitivity**: `format_sensitivity` (marked non-scoring in BFCL; optional).

Use **`--test-collection agentic_multi_turn`** (or `multi_turn` / `agentic` alone) to run only
those V4 tracks without enumerating every category. For HASHIRU **CEO + worker agents + tools**,
add **`--ceo-force-agents`** (official FC-string scores may drop; use traces / your semantic
metrics for worker evaluation).

What this script does:
1. Loads BFCL V4 entries via `bfcl_eval.utils.load_dataset_entry` (one or many categories).
2. Queries HASHIRU Gradio `/chat` through a custom `GorillaHandler`.
3. Writes BFCL JSONL results (`handler.write` groups by test id → category file).
4. Runs the official BFCL evaluator on the categories you ran.

**Interpreting very low `multi_turn_*` scores:** The official checker requires the same number of
outer conversational turns as the gold trace. BFCL caps inner steps per user turn (~20). If the
model spends those steps repeating bad tool calls on turn 1, the run never reaches later user
turns → `force_terminated` / turn-count mismatch and **0% accuracy** even when HASHIRU is
otherwise healthy. See `README_BFCL.md` in this folder.

**Agents / modes:** For `multi_turn_*` and `agentic_*` categories this runner uses **BFCL-direct**
mode (`modeIndexes=[]`): HASHIRU **CEO and worker agents are off** so each step returns one
Gorilla-parseable Python call, as the official BFCL harness requires. That is intentional, not a
misconfiguration. For agent benchmarking use e.g. `--test-category simple_python --ceo-force-agents`.

**Huge HASHIRU token counts:** Each inner step resends the full BFCL transcript including large
simulated `tool` JSON (`ls` in long-context). Optional `--summarize-long-bfcl-tool-json` trims tool
blobs in the Gradio prompt only (simulator state unchanged).

Prereqs:
  - pip install bfcl-eval gradio_client tqdm python-dotenv
  - HASHIRU Gradio server at --gradio_url (default http://127.0.0.1:7860/)
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional


# Mirrors bfcl_eval.constants.category_mapping (fallback if import lacks symbols).
_FALLBACK_MULTI_TURN_CATEGORY: list[str] = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]
_FALLBACK_AGENTIC_CATEGORY: list[str] = [
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
    "web_search_base",
    "web_search_no_snippet",
]


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _load_bfcl_v4_category_groups() -> tuple[list[str], list[str], dict[str, list[str]]]:
    """
    Return (multi_turn_categories, agentic_categories, test_collection_mapping_subset).

    Prefer live symbols from bfcl_eval; fall back to V4 lists aligned with Gorilla repo.
    """
    try:
        import bfcl_eval.constants.category_mapping as cm  # type: ignore

        mt = list(getattr(cm, "MULTI_TURN_CATEGORY", _FALLBACK_MULTI_TURN_CATEGORY))
        ag = list(getattr(cm, "AGENTIC_CATEGORY", _FALLBACK_AGENTIC_CATEGORY))
        mapping: dict[str, list[str]] = {}
        full_map = getattr(cm, "TEST_COLLECTION_MAPPING", None)
        if isinstance(full_map, dict):
            for key in ("multi_turn", "agentic", "agentic_multi_turn"):
                if key in full_map and isinstance(full_map[key], (list, tuple)):
                    mapping[key] = list(full_map[key])
        if "multi_turn" not in mapping:
            mapping["multi_turn"] = mt
        if "agentic" not in mapping:
            mapping["agentic"] = ag
        if "agentic_multi_turn" not in mapping:
            mapping["agentic_multi_turn"] = _dedupe_preserve_order(mt + ag)
        return mt, ag, mapping
    except Exception:
        mapping = {
            "multi_turn": list(_FALLBACK_MULTI_TURN_CATEGORY),
            "agentic": list(_FALLBACK_AGENTIC_CATEGORY),
            "agentic_multi_turn": _dedupe_preserve_order(
                _FALLBACK_MULTI_TURN_CATEGORY + _FALLBACK_AGENTIC_CATEGORY
            ),
        }
        return (
            list(_FALLBACK_MULTI_TURN_CATEGORY),
            list(_FALLBACK_AGENTIC_CATEGORY),
            mapping,
        )


def _expand_test_collection(name: str) -> list[str]:
    _, _, mapping = _load_bfcl_v4_category_groups()
    key = name.strip().lower().replace("-", "_")
    if key == "complex" or key == "agentic_multi_turn":
        return list(mapping.get("agentic_multi_turn", mapping["multi_turn"] + mapping["agentic"]))
    if key in mapping:
        return list(mapping[key])
    raise ValueError(f"Unknown test collection {name!r}")


def _bfcl_v4_multi_turn_and_agentic_ids() -> frozenset[str]:
    """Categories where bfcl_eval runs a multi-step FC protocol (must get Gorilla-parseable strings)."""
    mt, ag, _ = _load_bfcl_v4_category_groups()
    return frozenset(_dedupe_preserve_order(list(mt) + list(ag)))


def _resolve_planned_bfcl_categories(args: argparse.Namespace) -> list[str]:
    if args.test_collection:
        return _expand_test_collection(args.test_collection)
    return [args.test_category]


CEO_FORCE_AGENTS_PREFIX = (
    "IMPORTANT CEO INSTRUCTIONS:\n"
    "- You MUST use agents to solve this.\n"
    "- Do NOT answer directly without delegating.\n"
    "- After agents respond, provide ONLY the final answer.\n"
)

BFCL_EXECUTION_GUARDRAILS = (
    "BFCL EXECUTION RULES:\n"
    "- Solve this in one pass and terminate.\n"
    "- Do NOT call PythonSandboxTool for simple arithmetic.\n"
    "- Avoid repeated tool calls with the same arguments.\n"
    "- Create at most one agent only if strictly necessary.\n"
    "- If you already have enough information, return the final function call immediately.\n"
)

# BFCL official checker expects a *single python-style API call* matching the schema name,
# not natural language and not HASHIRU tool traces.
BFCL_DIRECT_PREFIX = (
    "BFCL / GORILLA FUNCTION-CALLING MODE (read carefully):\n"
    "- You are answering a Berkeley Function Calling Leaderboard item.\n"
    "- Do NOT call any tools (no PythonSandboxTool, no AskAgent, no GetAgents, no semantic metrics).\n"
    "- Do NOT delegate to worker agents. Respond as the API router in plain text only.\n"
    "- Read the user question and the BFCL TARGET API JSON below.\n"
    "- Output EXACTLY ONE line: a valid Python function call using the EXACT API name and "
    "keyword arguments (e.g. calculate_triangle_area(base=10, height=5), "
    "math.factorial(number=5), math.hypot(x=4, y=5)).\n"
    "- No markdown, no code fences, no explanation before or after the call.\n"
)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if "```" not in s:
        return s
    # Keep the first fenced block content.
    first = s.find("```")
    s2 = s[first + 3 :].lstrip()
    # Optional first-line language tag (python, json, …) — avoid regex that can
    # break across Python/re versions (e.g. nested "[:...]" character classes).
    if "\n" in s2:
        first_line, rest = s2.split("\n", 1)
        if re.match(r"^[A-Za-z0-9_+.=#-]{1,40}$", first_line.strip()):
            s2 = rest
    end = s2.find("```")
    if end != -1:
        s2 = s2[:end]
    return s2.strip()


def _strip_answer_markers(t: str) -> str:
    """Remove common 'FINAL ANSWER:' / 'Answer:' prefixes without regex groups."""
    tl = t.lower()
    for marker in (
        "final answer:",
        "final answer -",
        "final answer –",
        "answer:",
        "answer -",
        "answer –",
    ):
        i = tl.find(marker)
        if i >= 0:
            return t[i + len(marker) :].strip()
    return t.strip()


def _first_balanced_call(s: str) -> str:
    """Find first identifier( ... ) with parenthesis matching (handles nested calls)."""
    try:
        m = re.search(r"[A-Za-z_][\w.]*\s*\(", s)
    except re.error:
        m = None
    if not m:
        return s.strip()
    start = m.start()
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()
    return s[start:].strip()


def _is_non_fc_assistant_garbage(text: str) -> bool:
    """True if this looks like HASHIRU loop guards / API errors / empty-model messages, not an API call."""
    s = (text or "").strip().lower()
    if not s:
        return True
    prefixes = (
        "stopping due to",
        "no response from the model",
        "error generating response",
        "tokens used:",
    )
    if any(s.startswith(p) for p in prefixes):
        return True
    # Rate limits / transport errors must not be parsed as python calls.
    if "429" in s or "resource exhausted" in s or "resource_exhausted" in s.replace(" ", ""):
        return True
    return False


def _extract_fc_python_expression(text: str) -> str:
    """
    BFCL FC (Gorilla style) expects something that can be parsed as python expressions
    after wrapping with [] (see GorillaHandler.decode_execute()).

    So we try to isolate the function-call expression(s) from a messy model output.
    """
    if not text:
        return ""
    if _is_non_fc_assistant_garbage(text):
        return ""
    try:
        t = _strip_code_fences(str(text))
        t = _strip_answer_markers(t)

        # If the model already output an outer list, unwrap it.
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1].strip()

        expr = _first_balanced_call(t)
        expr = expr.rstrip(";\n ").strip()

        if expr.startswith("[") and expr.endswith("]"):
            expr = expr[1:-1].strip()
        return expr
    except re.error:
        # Never fail the benchmark run on regex edge cases; return best-effort text.
        return str(text).strip()


def _gorilla_fc_ast_ok(expr: str) -> bool:
    """
    True if wrapping expr as [...] matches bfcl_eval ast_parse (PYTHON) expectations:
    one Call or a list/tuple of Calls.
    """
    if not expr or not str(expr).strip():
        return False
    try:
        cleaned = str(expr).strip().strip("'").strip('"')
        wrapped = "[" + cleaned + "]"
        parsed = ast.parse(wrapped, mode="eval")
        body = parsed.body
        if isinstance(body, ast.Call):
            return True
        if isinstance(body, ast.List):
            return bool(body.elts) and all(isinstance(e, ast.Call) for e in body.elts)
        if isinstance(body, ast.Tuple):
            return bool(body.elts) and all(isinstance(e, ast.Call) for e in body.elts)
        return False
    except SyntaxError:
        return False


def _iter_fc_expression_candidates(text: str) -> list[str]:
    """Ordered candidates: whole message, then each non-empty line (bottom-up)."""
    if not text or _is_non_fc_assistant_garbage(text):
        return []
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        e = (s or "").strip()
        if not e or e in seen:
            return
        seen.add(e)
        out.append(e)

    add(_extract_fc_python_expression(text))
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        add(_extract_fc_python_expression(line))
    return out


def _safe_error_expression(msg: str) -> str:
    # Keep error outputs syntactically parseable by BFCL AST parser.
    clean = str(msg).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'__hashiru_error__(message="{clean}")'


def _get_last_assistant_content(history: Any) -> str:
    """
    Extract last assistant message content from gradio_client's chat history.
    """
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return ""

    for turn in reversed(history):
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            # Legacy gradio format: (user, bot)
            if turn[1]:
                return str(turn[1])
            continue
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "assistant":
            continue

        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            return content

        fr = turn.get("function_response") or {}
        out = (fr.get("result") or {}).get("output")
        if out:
            return str(out)

        # Some servers wrap content in parts.
        if isinstance(content, dict):
            parts = content.get("parts") or []
            if parts and isinstance(parts[0], dict) and parts[0].get("text"):
                return str(parts[0]["text"])

    return ""


def _iter_assistant_texts_newest_first(history: Any) -> list[str]:
    """Collect assistant string contents from newest to oldest."""
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return []
    out: list[str] = []

    def _push(msg: dict) -> None:
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            out.append(content)
            return
        fr = msg.get("function_response") or {}
        o = (fr.get("result") or {}).get("output")
        if o:
            out.append(str(o))
            return
        if isinstance(content, dict):
            parts = content.get("parts") or []
            if parts and isinstance(parts[0], dict) and parts[0].get("text"):
                out.append(str(parts[0]["text"]))

    for turn in reversed(history):
        if isinstance(turn, (list, tuple)) and len(turn) >= 2 and turn[1]:
            out.append(str(turn[1]))
            continue
        if isinstance(turn, dict):
            if turn.get("role") == "function_call":
                py = _function_call_payloads_to_python_line(turn.get("content"))
                if py:
                    out.append(py)
                continue
            if turn.get("role") == "assistant":
                _push(turn)
    return out


def _pick_content_for_bfcl_extraction(history: Any) -> str:
    """
    Prefer the newest assistant turn that yields a plausible FC snippet; avoids loop-guard text.
    """
    for txt in _iter_assistant_texts_newest_first(history):
        if _is_non_fc_assistant_garbage(txt):
            continue
        ex = _extract_fc_python_expression(txt)
        if ex and re.match(r"^[A-Za-z_][\w.]*\s*\(", ex.strip()):
            return txt
    # Fall back to legacy single-last-message behavior
    return _get_last_assistant_content(history)


def _format_bfcl_function_block(functions: Any) -> str:
    """Inject BFCL tool schema so HASHIRU outputs the checker-expected function name."""
    if not functions:
        return ""
    try:
        return (
            "\n\nBFCL TARGET API (use EXACTLY this function name; one line, no markdown):\n"
            + json.dumps(functions, indent=2, ensure_ascii=False)
        )
    except (TypeError, ValueError):
        return ""


def _collect_bfcl_schema_function_names(functions: Any) -> list[str]:
    """Top-level 'name' fields from BFCL function/tool list (for an explicit allow-list hint)."""
    if not isinstance(functions, list):
        return []
    return [str(item["name"]) for item in functions if isinstance(item, dict) and item.get("name")]


# Appended only from the benchmark (no HASHIRU app changes). Helps multi_turn sim tasks.
BFCL_MULTI_TURN_SIMULATOR_HINT = (
    "\n\nMULTI-TURN SIMULATOR RULES:\n"
    "- TOOL RESULT lines come from BFCL's simulated GorillaFileSystem/API, not your real machine.\n"
    "- If a call fails (e.g. \"No such file\"), change strategy: often you must cd(folder=...) into "
    "the directory the USER named *before* mv/cat/grep on paths there. Typical gold order: "
    "cd into the folder that contains the file, then mkdir if needed, then mv.\n"
    "- If the last TOOL RESULT shows success for an action (e.g. JSON with \"result\", \"moved\", "
    "\"removed\", or no \"error\" key after mv), that step is DONE — do NOT repeat the same "
    "mv/rm/touch/echo call; move on or stop until the next USER message appears in the transcript.\n"
    "- Do NOT repeat the identical failing call many times; read the error and try cd, ls, or pwd.\n"
    "- Do only what the latest USER message in the transcript asks; do not run grep/sort/diff/cp/tweet "
    "for a future user turn that is not yet in the conversation.\n"
    "- Use one python call per step, matching the BFCL JSON schema (parameter names and types).\n"
    "- For find(), include parameters the schema lists, usually path and name, e.g. "
    "find(path='.', name='test').\n"
    "- Do not invent nested names like find.find(...); use the single function name from the schema.\n"
)

# multi_turn_long_context adds many filler filenames to ls() output; models often loop ls() forever.
BFCL_LONG_CONTEXT_LISTING_HINT = (
    "\n\nLONG-CONTEXT DIRECTORY NOISE:\n"
    "- In multi_turn_long_context, directory listings may include many unrelated filenames. "
    "That is normal. Do NOT call ls() again if you already have the listing — pick the next action "
    "the USER asked for (cd into a named folder, sort a named file, post_tweet, etc.).\n"
    "- If the USER names a directory (tmp, document, workspace, documents, archive), cd(folder='...') "
    "there first when you need paths inside it; match the USER's spelling, or try close variants if cd fails.\n"
)


def _bfcl_benchmark_prompt_suffix(
    fn_block: str,
    fn_spec: Any,
    stashed: Optional[dict[str, Any]],
) -> str:
    """Extra instructions localized to the benchmark client (original HASHIRU unchanged)."""
    if not (fn_block and str(fn_block).strip()):
        return ""
    parts: list[str] = []
    names = _collect_bfcl_schema_function_names(fn_spec)
    if names:
        parts.append(
            "\n\nALLOWED TOP-LEVEL API NAMES (match spelling; add ClassName. prefix in the call if "
            "the schema shows it, e.g. GorillaFileSystem.cd):\n"
            + ", ".join(names)
        )
    tid = str(stashed.get("id", "")) if isinstance(stashed, dict) else ""
    if "multi_turn" in tid:
        parts.append(BFCL_MULTI_TURN_SIMULATOR_HINT)
    if "long_context" in tid:
        parts.append(BFCL_LONG_CONTEXT_LISTING_HINT)
    return "".join(parts)


def _summarize_bfcl_tool_content_for_prompt(content: str) -> str:
    """
    Shrink huge BFCL tool JSON (long_context ls listings, cat/grep payloads) for the HASHIRU prompt
    only. Does not change bfcl_eval's simulator — only what we send to Gradio.
    """
    if not isinstance(content, str):
        return content
    s = content.strip()
    if not s or s == "None":
        return content
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return content
    if not isinstance(obj, dict):
        return content
    out = dict(obj)
    changed = False

    cdc = out.get("current_directory_content")
    if isinstance(cdc, list) and len(cdc) > 16:
        n_extra = len(cdc) - 14
        out["current_directory_content"] = list(cdc[:14])
        out["_bfcl_prompt_note"] = (
            f"{n_extra} more names omitted (BFCL long-context filler). "
            "Do NOT repeat ls(); follow the USER's next instruction (cd/mv/cp/sort/etc.)."
        )
        changed = True

    fc = out.get("file_content")
    if isinstance(fc, str) and len(fc) > 4000:
        out["file_content"] = fc[:4000] + "\n...[truncated in HASHIRU prompt only]..."
        changed = True

    ml = out.get("matching_lines")
    if isinstance(ml, list) and ml:
        total = sum(len(str(x)) for x in ml)
        if total > 6000:
            short: list[Any] = []
            budget = 5000
            for line in ml:
                t = str(line)
                if len(t) > 800:
                    t = t[:800] + "…[truncated]"
                if budget <= 0:
                    break
                short.append(t)
                budget -= len(t)
            out["matching_lines"] = short
            out["_bfcl_matching_lines_truncated"] = True
            changed = True

    ll = out.get("last_lines")
    if isinstance(ll, str) and len(ll) > 4000:
        out["last_lines"] = ll[:4000] + "\n...[truncated]..."
        changed = True

    if not changed:
        return content
    try:
        return json.dumps(out, ensure_ascii=False)
    except (TypeError, ValueError):
        return content


def _summarize_bfcl_messages_for_hashiru_prompt(messages: list[dict]) -> list[dict]:
    """Deep copy + summarize tool role strings so HASHIRU sees smaller prompts."""
    out = copy.deepcopy(messages)
    for m in out:
        if not isinstance(m, dict):
            continue
        if (m.get("role") or "").strip().lower() != "tool":
            continue
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = _summarize_bfcl_tool_content_for_prompt(c)
    return out


def _stringify_bfcl_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(content)


def _function_call_payloads_to_python_line(content: Any) -> str:
    """
    HASHIRU may record Gemini tool use as role=function_call with serialized payloads.
    Convert to a Gorilla-style python call line for BFCL decode_execute.
    """
    if not isinstance(content, list):
        return ""
    segments: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("kind") != "function_call":
            continue
        name = item.get("name") or ""
        args = item.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not name or not isinstance(args, dict):
            continue
        kw = ",".join(f"{k}={repr(v)}" for k, v in args.items())
        segments.append(f"{name}({kw})")
    if not segments:
        return ""
    if len(segments) == 1:
        return segments[0]
    return ", ".join(segments)


def build_bfcl_fc_prompt_from_messages(
    messages: list[dict],
    extra_output_instruction: str,
    *,
    leader_prefix: str,
) -> str:
    """
    Serialize the full bfcl_eval FC transcript (system, user, assistant, tool).

    Multi-turn / long-context BFCL builds inference_data['message'] with assistant + tool
    turns after each simulated execution. Older code only forwarded user+system text, so
    HASHIRU had no prior-call / tool-output context → empty or unparseable replies and
    'Failed to decode the model response' in the harness.
    """
    system_chunks: list[str] = []
    dialogue_lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        text = _stringify_bfcl_message_content(m.get("content")).strip()
        if role == "system":
            if text:
                system_chunks.append(text)
        elif role == "user":
            if text:
                dialogue_lines.append("USER:\n" + text)
        elif role == "assistant":
            if text:
                dialogue_lines.append("ASSISTANT (your prior API call line):\n" + text)
        elif role == "tool":
            if text:
                dialogue_lines.append("TOOL RESULT (simulated environment; use for the next step):\n" + text)
        elif role == "function_call":
            py_line = _function_call_payloads_to_python_line(m.get("content"))
            if py_line:
                dialogue_lines.append("ASSISTANT (tool request as python call):\n" + py_line)
            elif text:
                dialogue_lines.append("ASSISTANT (tool metadata):\n" + text)

    blocks: list[str] = []
    if system_chunks:
        blocks.append("SYSTEM CONTEXT:\n" + "\n\n".join(system_chunks))
    if dialogue_lines:
        blocks.append(
            "CONVERSATION (multi-turn BFCL — follow order; your NEXT reply = ONE python call line only):\n\n"
            + "\n\n".join(dialogue_lines)
        )
    body = "\n\n".join(blocks).strip()
    prompt = (leader_prefix.rstrip() + "\n\n" + body + "\n\n" + extra_output_instruction).strip()
    return prompt


def build_bfcl_direct_prompt(messages: list[dict], extra_output_instruction: str) -> str:
    """Single-line FC prompt; includes full assistant/tool transcript for multi-turn."""
    return build_bfcl_fc_prompt_from_messages(
        messages,
        extra_output_instruction,
        leader_prefix=BFCL_DIRECT_PREFIX,
    )


def build_hashiru_prompt_from_bfcl_messages(messages: list[dict], extra_output_instruction: str) -> str:
    return build_bfcl_fc_prompt_from_messages(
        messages,
        extra_output_instruction,
        leader_prefix=CEO_FORCE_AGENTS_PREFIX + "\n\n" + BFCL_EXECUTION_GUARDRAILS + "\n\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gorilla BFCL using HASHIRU as the FC model.")
    parser.add_argument(
        "--gradio_url",
        type=str,
        default=os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860"),
        help="Base URL for HASHIRU Gradio app.",
    )
    parser.add_argument(
        "--test-category",
        type=str,
        default="simple_python",
        help=(
            "Single BFCL V4 category (e.g. simple_python, multi_turn_base, memory_kv). "
            "Ignored if --test-collection is set."
        ),
    )
    parser.add_argument(
        "--test-collection",
        type=str,
        default=None,
        choices=("multi_turn", "agentic", "agentic_multi_turn", "complex"),
        help=(
            "Run a predefined V4 group instead of --test-category: "
            "'multi_turn' = multi_turn_base, miss_func, miss_param, long_context; "
            "'agentic' = memory_kv, memory_vector, memory_rec_sum, web_search_base, web_search_no_snippet; "
            "'agentic_multi_turn' or 'complex' = union of both (recommended for agentic / multi-step focus)."
        ),
    )
    parser.add_argument(
        "--list-bfcl-focus-categories",
        action="store_true",
        help="Print BFCL V4 multi-turn and agentic category IDs (from bfcl_eval when available) and exit.",
    )
    parser.add_argument(
        "--max-test-cases",
        type=int,
        default=None,
        help="Limit how many BFCL entries to run (if set smaller than the full category, BFCL runs evaluation in partial-eval mode).",
    )
    parser.add_argument(
        "--include-input-log",
        action="store_true",
        default=False,
        help="If set, include bfcl_eval inference input logs (can make results files large).",
    )
    parser.add_argument(
        "--exclude-state-log",
        action="store_true",
        default=True,
        help="Exclude state logs (default True).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        default=False,
        help="If set, update existing BFCL result entries instead of appending.",
    )
    parser.add_argument(
        "--bfcl-project-root",
        type=str,
        default=None,
        help="Optional override for where BFCL writes result/score folders (sets BFCL_PROJECT_ROOT).",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=180,
        help="Per-test-case timeout. Use 240+ with --ceo-force-agents (slow).",
    )
    parser.add_argument(
        "--chat-retries",
        type=int,
        default=2,
        help="Retries per case when /chat fails or returns empty content.",
    )
    parser.add_argument(
        "--ceo-force-agents",
        action="store_true",
        help=(
            "Run HASHIRU with CEO agents + tools enabled. Default is BFCL-direct mode: no agents/tools, "
            "single-line API call only (matches what the BFCL evaluator checks). "
            "Ignored for multi-turn / agentic BFCL categories unless --unsafe-ceo-with-bfcl-complex."
        ),
    )
    parser.add_argument(
        "--ceo-no-dynamic-tools",
        action="store_true",
        help=(
            "With --ceo-force-agents: enable agents + tool invocation but NOT ToolCreator "
            "(avoids mkdir/mv codegen loops; BFCL-style APIs should be answered as FC text, not new tools)."
        ),
    )
    parser.add_argument(
        "--unsafe-ceo-with-bfcl-complex",
        action="store_true",
        help=(
            "Allow --ceo-force-agents together with multi_turn_* / memory_* / web_search_* categories. "
            "bfcl_eval will often print 'Failed to decode the model response' and burn tokens — "
            "only for debugging."
        ),
    )
    parser.add_argument(
        "--bfcl-backoff-seconds",
        type=float,
        default=25.0,
        help="After a 429 / RESOURCE_EXHAUSTED from Gradio, wait this many seconds before retrying a prompt.",
    )
    parser.add_argument(
        "--inter-case-sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between BFCL test cases to reduce Gemini rate limits.",
    )
    parser.add_argument(
        "--bfcl-max-inner-steps",
        type=int,
        default=0,
        help=(
            "If > 0, patch bfcl_eval's MAXIMUM_STEP_LIMIT to this value for the process (unofficial; "
            "default 0 leaves BFCL's built-in ~20 step cap). Helps when the model needs more inner "
            "steps per user turn on multi_turn_long_context."
        ),
    )
    parser.add_argument(
        "--verbose-bfcl",
        action="store_true",
        help=(
            "Log each HASHIRU FC step to stderr: test id, approximate prompt size, extracted "
            "function-call line (helps when bfcl_eval only prints Step: N / Gradio loaded)."
        ),
    )
    parser.add_argument(
        "--summarize-long-bfcl-tool-json",
        action="store_true",
        help=(
            "Truncate huge simulated TOOL RESULT JSON in prompts sent to HASHIRU only (long_context "
            "ls listings, cat/grep blobs). BFCL's in-process simulator still uses full fidelity."
        ),
    )
    args = parser.parse_args()

    if args.list_bfcl_focus_categories:
        mt, ag, mapping = _load_bfcl_v4_category_groups()
        print("BFCL V4 — categories aimed at multi-turn & agentic evaluation:\n")
        print("MULTI_TURN_CATEGORY:")
        for c in mt:
            print(f"  - {c}")
        print("\nAGENTIC_CATEGORY (memory + web search):")
        for c in ag:
            print(f"  - {c}")
        print("\nCLI shortcuts (--test-collection):")
        print(f"  multi_turn          -> {mapping.get('multi_turn', mt)}")
        print(f"  agentic             -> {mapping.get('agentic', ag)}")
        print(
            "  agentic_multi_turn  -> "
            f"{mapping.get('agentic_multi_turn', _dedupe_preserve_order(mt + ag))}"
        )
        print("  complex             -> same as agentic_multi_turn")
        print(
            "\nNote: V4 also defines `format_sensitivity` (separate robustness track; "
            "often treated as non-scoring in leaderboard aggregation). "
            "Run it with: --test-category format_sensitivity"
        )
        print(
            "\nOther V4 tracks (single-turn AST, live, etc.) are listed in "
            "bfcl_eval.constants.category_mapping — use --test-category for those."
        )
        print(
            "\nCEO / agents: multi-turn + agentic BFCL categories require Gorilla FC text each step. "
            "This runner disables --ceo-force-agents for those unless --unsafe-ceo-with-bfcl-complex."
        )
        raise SystemExit(0)

    # Ensure we write results into a project folder under this repo.
    if args.bfcl_project_root:
        project_root = Path(args.bfcl_project_root).expanduser().resolve()
    else:
        # Put it under HASHIRU_Bench/bench/results/bfcl/<timestamp>
        ts = time.strftime("%Y%m%d_%H%M%S")
        project_root = (
            Path(__file__).resolve().parent.parent / "results" / "bfcl" / f"bfcl_{ts}"
        ).resolve()
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)

    try:
        from gradio_client import Client
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: gradio_client. Install with: pip install gradio_client"
        ) from e

    try:
        from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH, DOTENV_PATH
        from bfcl_eval.utils import load_dataset_entry
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX
        from bfcl_eval.model_handler.api_inference.gorilla import GorillaHandler
        from bfcl_eval.eval_checker.eval_runner import main as bfcl_eval_main
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: bfcl_eval. Install with: pip install bfcl-eval"
        ) from e

    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    if "gorilla-openfunctions-v2" not in MODEL_CONFIG_MAPPING:
        raise SystemExit("bfcl-eval does not contain 'gorilla-openfunctions-v2' in MODEL_CONFIG_MAPPING.")

    if int(getattr(args, "bfcl_max_inner_steps", 0) or 0) > 0:
        try:
            import bfcl_eval.constants.default_prompts as _bfcl_dp  # type: ignore

            _old_lim = getattr(_bfcl_dp, "MAXIMUM_STEP_LIMIT", None)
            _bfcl_dp.MAXIMUM_STEP_LIMIT = int(args.bfcl_max_inner_steps)
            print(
                f"Note: patched bfcl_eval MAXIMUM_STEP_LIMIT {_old_lim!r} -> {_bfcl_dp.MAXIMUM_STEP_LIMIT} "
                "(unofficial; leaderboard comparisons use the default cap)."
            )
        except Exception as _e:
            print(f"Warning: could not patch MAXIMUM_STEP_LIMIT: {_e}", file=sys.stderr)

    planned_categories = _resolve_planned_bfcl_categories(args)
    complex_bfcl_ids = _bfcl_v4_multi_turn_and_agentic_ids()
    touches_complex_bfcl = bool(complex_bfcl_ids.intersection(planned_categories))

    effective_ceo_agents = bool(args.ceo_force_agents)
    if effective_ceo_agents and touches_complex_bfcl and not args.unsafe_ceo_with_bfcl_complex:
        effective_ceo_agents = False
        print(
            "\n*** BFCL: disabling --ceo-force-agents for this run ***\n"
            "Multi-turn / agentic BFCL categories (multi_turn_*, memory_*, web_search_*) use a harness "
            "that must decode a Gorilla function-call string at every step. CEO + ToolCreator + agents "
            "returns tool calls / prose → 'Failed to decode the model response' and long loops.\n"
            "Running in BFCL-direct mode (empty modeIndexes) so the official benchmark can complete.\n"
            "For CEO+worker experiments, use single-turn --test-category (e.g. simple_python) with "
            "--ceo-force-agents, or pass --unsafe-ceo-with-bfcl-complex to force CEO here anyway.\n"
        )
    elif effective_ceo_agents and touches_complex_bfcl and args.unsafe_ceo_with_bfcl_complex:
        print(
            "\n*** WARNING: --unsafe-ceo-with-bfcl-complex — expect BFCL decode failures and high token use. ***\n"
        )

    if touches_complex_bfcl and not effective_ceo_agents:
        print(
            "\n=== BFCL multi-step mode (intended configuration) ===\n"
            "- Worker agents / CEO orchestration: OFF (empty Gradio modeIndexes).\n"
            "- HASHIRU acts as a single LLM that must print one Python function call per inner step.\n"
            "- If HASHIRU shows 'Selected modes: []' and huge input tokens, that matches this mode.\n"
            "- For agent+tool benchmarking, use e.g. --test-category simple_python --ceo-force-agents.\n"
            "- Optional: --verbose-bfcl (stderr), --summarize-long-bfcl-tool-json (smaller prompts).\n"
            "======================================================\n"
        )

    if args.ceo_no_dynamic_tools and not args.ceo_force_agents:
        print("Note: --ceo-no-dynamic-tools has no effect without --ceo-force-agents.", file=sys.stderr)

    # BFCL output expects only a python function call expression (no wrappers).
    extra_output_instruction = (
        "OUTPUT INSTRUCTION:\n"
        "Return ONLY the function-call expression(s) in python syntax "
        "(e.g., func_name(arg=...)). No extra text.\n"
    )

    # Default BFCL run: empty mode list => no agent creation, no tools (text-only FC matches BFCL scorer).
    if effective_ceo_agents:
        mode_indexes = [
            "ENABLE_AGENT_CREATION",
            "ENABLE_LOCAL_AGENTS",
        ]
        if not args.ceo_no_dynamic_tools:
            mode_indexes.append("ENABLE_TOOL_CREATION")
        mode_indexes.append("ENABLE_TOOL_INVOCATION")
    else:
        mode_indexes = []

    class HashiruBFCLGorillaHandler(GorillaHandler):
        def __init__(
            self,
            gradio_url: str,
            mode_indexes: list[str],
            ceo_prefix: str,
            bfcl_direct: bool,
            bfcl_backoff_seconds: float = 25.0,
            *,
            verbose_bfcl: bool = False,
            summarize_tool_json: bool = False,
            **kwargs: Any,
        ) -> None:
            chat_retries = max(1, int(kwargs.pop("chat_retries", 2)))
            # Force gorilla-openfunctions-v2 decoding expectations.
            super().__init__(
                model_name="gorilla-openfunctions-v2",
                temperature=0,
                registry_name="gorilla-openfunctions-v2",
                is_fc_model=True,
                **kwargs,
            )
            self._gradio_url = gradio_url.rstrip("/") + "/"
            self._mode_indexes = mode_indexes
            self._ceo_prefix = ceo_prefix
            self._bfcl_direct = bfcl_direct
            self._bfcl_backoff_seconds = max(0.0, float(bfcl_backoff_seconds))
            self._bfcl_stashed_entry: Optional[dict[str, Any]] = None
            self._chat_retries = chat_retries
            self._verbose_bfcl = bool(verbose_bfcl)
            self._summarize_tool_json = bool(summarize_tool_json)

        def inference(self, test_entry, *args: Any, **kwargs: Any):
            """
            bfcl_eval's Gorilla path often omits `function` from inference_data passed to _query_FC.
            Stash the full test entry so we can still inject BFCL TARGET API JSON.
            """
            self._bfcl_stashed_entry = test_entry
            try:
                return super().inference(test_entry, *args, **kwargs)
            finally:
                self._bfcl_stashed_entry = None

        def _predict_chat_isolated(self, prompt: str) -> tuple[Any, Any]:
            """
            New Gradio client + single-turn chat so HASHIRU does not accumulate BFCL cases
            in one session (that caused huge prompts, 429 RESOURCE_EXHAUSTED, and stale
            answers being picked for later items).
            """
            client = Client(self._gradio_url)
            client.predict(modeIndexes=self._mode_indexes, api_name="/update_model")
            return client.predict(
                {"text": prompt, "files": []},
                None,
                api_name="/chat",
            )

        def _query_FC(self, inference_data: dict):
            # Build prompt from BFCL's message format.
            messages = inference_data.get("message") or []
            if not isinstance(messages, list):
                messages = []
            if self._summarize_tool_json:
                messages = _summarize_bfcl_messages_for_hashiru_prompt(messages)
            # bfcl_eval stores schemas under "tools" after _compile_tools; "function" is often unset.
            fn_spec = inference_data.get("tools") or inference_data.get("function")
            if not fn_spec and isinstance(self._bfcl_stashed_entry, dict):
                fn_spec = self._bfcl_stashed_entry.get("function")
            fn_block = _format_bfcl_function_block(fn_spec)
            bench_suffix = _bfcl_benchmark_prompt_suffix(
                fn_block, fn_spec, self._bfcl_stashed_entry
            )
            if self._bfcl_direct:
                strict_fc = (
                    "\n\nFINAL CHECK: Your entire reply must be ONE line — the python call only."
                    if fn_block
                    else ""
                )
                prompt = build_bfcl_direct_prompt(
                    messages=messages,
                    extra_output_instruction=extra_output_instruction,
                )
                prompt = (prompt.strip() + fn_block + bench_suffix + strict_fc).strip()
                user_blob = "\n".join(
                    _stringify_bfcl_message_content(m.get("content"))
                    for m in messages
                    if isinstance(m, dict) and _stringify_bfcl_message_content(m.get("content")).strip()
                )
                fallback_prompt = (
                    "Return ONLY one valid python function call for this task.\n"
                    "Use the EXACT function name from BFCL TARGET API. Keyword arguments only. One line.\n\n"
                    + user_blob
                    + "\n\n"
                    + extra_output_instruction
                    + fn_block
                    + bench_suffix
                    + strict_fc
                ).strip()
            else:
                strict_fc = (
                    "\n\nCRITICAL: After any brief agent step, your final message must be ONE python "
                    "function call using the EXACT name from BFCL TARGET API (e.g. math.factorial(number=5), "
                    "geometry.area_circle(radius=10)). No PythonSandboxTool, no repeated tools, no prose."
                    if fn_block
                    else ""
                )
                prompt = build_hashiru_prompt_from_bfcl_messages(
                    messages=messages,
                    extra_output_instruction=extra_output_instruction,
                )
                prompt = prompt.replace(CEO_FORCE_AGENTS_PREFIX, self._ceo_prefix)
                prompt = (prompt.strip() + fn_block + bench_suffix + strict_fc).strip()
                fallback_prompt = (
                    "Return ONLY one valid python function call expression for the request.\n"
                    "Do not include prose or markdown.\n\n"
                    + "\n".join(
                        _stringify_bfcl_message_content(m.get("content"))
                        for m in messages
                        if isinstance(m, dict)
                        and _stringify_bfcl_message_content(m.get("content")).strip()
                    )
                    + "\n\n"
                    + extra_output_instruction
                    + fn_block
                    + bench_suffix
                    + strict_fc
                ).strip()

            t0 = time.time()
            fc_expr = ""
            last_err = ""

            prompts = [prompt, fallback_prompt]
            for i in range(self._chat_retries):
                try_prompt = prompts[min(i, len(prompts) - 1)]
                try:
                    response, history = self._predict_chat_isolated(try_prompt)
                    content = _pick_content_for_bfcl_extraction(history)
                    if not content and response:
                        content = str(response)
                    fc_expr = ""
                    for cand in _iter_fc_expression_candidates(content):
                        if _gorilla_fc_ast_ok(cand):
                            fc_expr = cand
                            break
                    if not fc_expr:
                        fallback = _extract_fc_python_expression(content)
                        if _gorilla_fc_ast_ok(fallback):
                            fc_expr = fallback
                    if fc_expr:
                        break
                    last_err = "empty_or_unparseable_response"
                except Exception as e:
                    err_txt = f"{type(e).__name__}: {e}"
                    last_err = err_txt
                    if "429" in err_txt or "RESOURCE_EXHAUSTED" in err_txt:
                        if self._bfcl_backoff_seconds > 0:
                            time.sleep(self._bfcl_backoff_seconds)

            if not fc_expr:
                fc_expr = _safe_error_expression(last_err or "no_function_call_extracted")
            latency = time.time() - t0

            if self._verbose_bfcl:
                tid = (
                    (self._bfcl_stashed_entry or {}).get("id", "?")
                    if isinstance(self._bfcl_stashed_entry, dict)
                    else "?"
                )
                print(
                    f"  [BFCL→HASHIRU] id={tid} prompt_chars≈{len(prompt)} msgs={len(messages)} "
                    f"fc={fc_expr[:180]!r}{'…' if len(fc_expr) > 180 else ''}",
                    file=sys.stderr,
                )

            # Return a fake response in the shape GorillaHandler._parse_query_response_FC expects.
            api_response = {
                "choices": [
                    {
                        "message": {
                            "content": fc_expr,
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }
            return api_response, latency

        def decode_execute(self, result, has_tool_call_tag):
            """
            bfcl_eval GorillaHandler historically called ast_parse(func, has_tool_call_tag) with two
            positionals; current ast_parse is (input_str, language=PYTHON, has_tool_call_tag=False),
            so False was interpreted as language → NotImplementedError / failed decode for every turn.
            """
            from inspect import signature

            from bfcl_eval.model_handler.utils import ast_parse

            func = "[" + (result or "").strip() + "]"
            sig = signature(ast_parse)
            if "language" in sig.parameters:
                from bfcl_eval.constants.enums import ReturnFormat

                # Prefer keywords so newer ast_parse signatures cannot mis-bind positionals.
                try:
                    decoded_output = ast_parse(
                        func,
                        language=ReturnFormat.PYTHON,
                        has_tool_call_tag=has_tool_call_tag,
                    )
                except TypeError:
                    decoded_output = ast_parse(
                        func, ReturnFormat.PYTHON, has_tool_call_tag
                    )
            else:
                decoded_output = ast_parse(func, has_tool_call_tag)
            execution_list = []
            for function_call in decoded_output:
                for key, value in function_call.items():
                    execution_list.append(
                        f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                    )
            return execution_list

        def _parse_query_response_FC(self, api_response: Any) -> dict:
            # Use GorillaHandler's logic for the response schema.
            return super()._parse_query_response_FC(api_response)

    def _make_bfcl_handler() -> HashiruBFCLGorillaHandler:
        """Fresh handler per test case avoids any instance-level carryover across BFCL ids."""
        return HashiruBFCLGorillaHandler(
            gradio_url=args.gradio_url,
            mode_indexes=mode_indexes,
            ceo_prefix=CEO_FORCE_AGENTS_PREFIX,
            bfcl_direct=not effective_ceo_agents,
            bfcl_backoff_seconds=args.bfcl_backoff_seconds,
            chat_retries=args.chat_retries,
            verbose_bfcl=bool(args.verbose_bfcl),
            summarize_tool_json=bool(args.summarize_long_bfcl_tool_json),
        )

    if effective_ceo_agents:
        extra = " (ToolCreator disabled)" if args.ceo_no_dynamic_tools else ""
        print(f"BFCL mode: CEO + agents + tools{extra} (orchestration). Scores may stay low vs official FC format.")
    else:
        print(
            "BFCL mode: direct function-call text (default). "
            "Agents/tools disabled via empty modeIndexes for this run."
        )
        print(
            "Using isolated Gradio client per BFCL case (avoids chat history buildup and 429s). "
            f"429 backoff={args.bfcl_backoff_seconds}s, inter-case sleep={args.inter_case_sleep}s."
        )

    # Same category list as used for CEO/direct decision (above).
    test_categories = planned_categories
    if args.test_collection:
        print(
            f"Using --test-collection={args.test_collection!r} "
            f"({len(test_categories)} categories): {', '.join(test_categories)}"
        )

    # Multi-turn / agentic cases run many in-process steps per test id; default 180s is often too low.
    effective_case_timeout = max(1, int(args.case_timeout_seconds))
    if _bfcl_v4_multi_turn_and_agentic_ids().intersection(set(test_categories)):
        bumped = max(effective_case_timeout, 600)
        if bumped > effective_case_timeout:
            print(
                f"Note: multi-turn/agentic categories — raising --case-timeout-seconds from "
                f"{effective_case_timeout} to {bumped} (override with a larger explicit value)."
            )
            effective_case_timeout = bumped

    if any(str(c).startswith("multi_turn") for c in test_categories):
        print(
            "Note: BFCL multi-turn compares outer-turn counts to gold; an inner ~20-step cap per "
            "user turn can yield 0% if the model stalls on the first turn. See bench/README_BFCL.md. "
            "Optional: --bfcl-max-inner-steps 40 (unofficial) gives more steps per user turn."
        )

    results_to_write: list[dict[str, Any]] = []
    any_partial_eval = False
    global_idx = 0
    categories_ran: list[str] = []

    print(f"Writing BFCL results to: {RESULT_PATH} (BFCL_PROJECT_ROOT={project_root})")

    for cat in test_categories:
        full_entries = load_dataset_entry(cat)
        full_count = len(full_entries)

        test_entries = full_entries
        if args.max_test_cases is not None and args.max_test_cases > 0:
            test_entries = test_entries[: args.max_test_cases]

        if not test_entries:
            print(f"Warning: no entries for category {cat!r}, skipping.", file=sys.stderr)
            continue

        categories_ran.append(cat)

        if len(test_entries) != full_count:
            any_partial_eval = True

        print(
            f"Category {cat!r}: loaded {len(test_entries)}/{full_count} BFCL entries "
            f"(max_test_cases={args.max_test_cases})."
        )

        for test_case in test_entries:
            global_idx += 1
            test_id = test_case.get("id", "unknown")
            print(f"[{global_idx}] {cat} test_id={test_id} ...")

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        _make_bfcl_handler().inference,
                        test_case,
                        bool(args.include_input_log),
                        bool(args.exclude_state_log),
                    )
                    model_responses, metadata = fut.result(timeout=effective_case_timeout)
            except concurrent.futures.TimeoutError:
                model_responses = _safe_error_expression(
                    f"TimeoutError: case exceeded {effective_case_timeout}s timeout"
                )
                metadata = {
                    "latency": effective_case_timeout,
                    "input_token_count": 0,
                    "output_token_count": 0,
                }
            except Exception as e:
                model_responses = _safe_error_expression(f"{type(e).__name__}: {e}")
                metadata = {"latency": 0, "input_token_count": 0, "output_token_count": 0}

            entry = {"id": test_id, "result": model_responses, **metadata}
            results_to_write.append(entry)
            if args.inter_case_sleep > 0:
                time.sleep(args.inter_case_sleep)

    if not results_to_write or not categories_ran:
        raise SystemExit("No BFCL entries were run (empty categories or all skipped).")

    # Write all entries to BFCL result files (grouped by id → per-category JSONL).
    _make_bfcl_handler().write(
        results_to_write,
        result_dir=RESULT_PATH,
        update_mode=bool(args.allow_overwrite),
    )

    # Run official evaluation on the generated files for every category we touched.
    print("Running BFCL official evaluation ...")
    bfcl_eval_main(
        model=["gorilla-openfunctions-v2"],
        test_categories=categories_ran,
        result_dir=None,
        score_dir=None,
        partial_eval=any_partial_eval,
    )

    print(f"Done. Score files are under: {SCORE_PATH}")
    if DOTENV_PATH.exists():
        print(f"Note: BFCL .env located at: {DOTENV_PATH}")


if __name__ == "__main__":
    main()

