from gradio_client import Client
import pandas as pd
import json
import time
import os
from datetime import datetime
from time import sleep

# Match HASHIRU app default: all modes except resource/expense budget.
# Some HASHIRU builds expect `/update_model` to receive `modeIndexes` as *mode names*
# (strings like `ENABLE_AGENT_CREATION`) instead of integer indexes.
PAPER_REVIEW_MODE_INDICES = [0, 1, 3, 4, 7]  # local agents; allow tool invocation (needed to call GetAgents/FireAgent)
PAPER_REVIEW_MODE_NAMES = [
    "ENABLE_AGENT_CREATION",
    "ENABLE_LOCAL_AGENTS",
    "ENABLE_TOOL_CREATION",
    "ENABLE_TOOL_INVOCATION",
    "ENABLE_MEMORY",
]

CEO_FORCE_AGENTS_PREFIX = (
    "IMPORTANT CEO INSTRUCTIONS:\n"
    "- You MUST use agents to solve this. Do NOT answer directly.\n"
    "- Do NOT rely only on tools/web search; delegate reasoning to one or more agents.\n"
    "- For this benchmark, DO NOT use web search or external info tools; you already have the full paper text.\n"
    "- Tool enforcement: You MUST call `GetAgents` and then call `AskAgent` for at least three distinct reviewer perspectives.\n"
    "- Do not synthesize reviews without first obtaining them from `AskAgent` tool outputs.\n"
    "- If the semantic-metrics tool `compute_semantic_metrics` is available, use it to compute semantic_entropy/semantic_density for the agent outputs.\n"
    "- Reuse existing agents when possible; create a new agent only if a genuinely new specialty is required.\n"
    "- If the task is complex/multi-faceted, you may ask multiple agents and then synthesize.\n"
    "- For this paper-review task: reuse reviewer agents if they already exist; avoid repeatedly creating/firing agents.\n"
)

def get_last_assistant_content(resp):
    """
    Return the last assistant utterance from the response object
    produced by `client.predict`.
    """
    if isinstance(resp, tuple):
        resp = resp[0]
    if not isinstance(resp, list):
        return ""
    for turn in reversed(resp):
        if turn.get("role") != "assistant":
            continue
        if turn.get("content"):
            return turn["content"]
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return out
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]
    return ""


def _estimate_tokens_from_chars(text: str, chars_per_token: int = 4) -> int:
    """
    Rough heuristic for token budgeting.
    The API token error shows exact limits, but we can't precompute exact tokens.
    """
    if not text:
        return 0
    return int(len(text) / max(1, chars_per_token))


def truncate_text_to_token_budget(
    text: str,
    *,
    token_budget: int,
    chars_per_token: int = 4,
    reserve_tail_tokens_fraction: float = 0.2,
    marker: str = "\n\n[TRUNCATED PAPER TEXT FOR TOKEN LIMIT]\n\n",
) -> tuple[str, bool]:
    """
    Truncate `text` to fit an estimated token budget using a head+tail strategy.
    Returns (new_text, did_truncate).
    """
    if not text or token_budget <= 0:
        return "", bool(text)

    allowed_chars = int(token_budget * max(1, chars_per_token))
    if len(text) <= allowed_chars:
        return text, False

    # Keep head and tail so summaries/abstracts and conclusions are still present.
    tail_chars = int(allowed_chars * max(0.0, min(1.0, reserve_tail_tokens_fraction)))
    head_chars = max(0, allowed_chars - tail_chars - len(marker))

    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    return head + marker + tail, True

def benchmark_paper_reviews(
    csv_path,
    id_col="ID",
    text_col="concatenated_text",
    num_samples=None,
    offset=0,
    output_dir="results",
    max_continue_rounds=10,
    continue_sleep_seconds=2,
    max_input_tokens=1048575,
    chars_per_token=4,
    reserve_history_tokens=5000,
):
    """
    Benchmark agent performance on paper reviews.

    Args:
        csv_path: path to the pipe-separated CSV of papers + existing reviews
        id_col:    name of the column containing unique paper IDs
        text_col:  name of the column containing the full paper text
        num_samples: if set, randomly sample this many papers
        output_dir: where to write the JSONL results
        max_continue_rounds: max "continue" prompts waiting for FINAL DECISION (then save partial result)
        max_input_tokens: model input token hard limit (from API error message)
        chars_per_token: heuristic for budgeting (used to decide how much paper text to keep)
        reserve_history_tokens: extra headroom to account for conversation history growth on "continue"
    """
    # load CSV
    df = pd.read_csv(csv_path, sep="|")
    if offset or num_samples:
        end = offset + num_samples if num_samples else None
        df = df.iloc[offset:end].reset_index(drop=True)
    # prepare output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"paper_review_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")

    # init client — do NOT enable resource/economy budget (that blocks AskAgent after few steps)
    client = Client("http://127.0.0.1:7860/")
    try:
        # Prefer indices first: original server maps modeIndexes i -> Mode(i+1).
        client.predict(modeIndexes=PAPER_REVIEW_MODE_INDICES, api_name="/update_model")
    except Exception as e_indices:
        try:
            # Fallback: mode names (some HASHIRU builds expect strings in modeIndexes).
            client.predict(modeIndexes=PAPER_REVIEW_MODE_NAMES, api_name="/update_model")
        except Exception as e_names:
            print(
                "Warning: could not set modes via /update_model. "
                f"indices_error={e_indices}; names_error={e_names}; using server defaults."
            )


    results = []
    for idx, row in df.iterrows():
        paper_id = row[id_col]
        title = row["Title"]
        paper_text = row[text_col]

        # Build prompt head without paper text so we can reserve token budget.
        prompt_head = (
            CEO_FORCE_AGENTS_PREFIX
            + "\nObtain three distinct reviewer perspectives by calling the `AskAgent` tool on three (preferably existing) reviewer agents. "
            + "Each reviewer should provide a review of the paper and recommend Accept/Reject for ICLR 2023. "
            + "The review should be detailed and include strengths and weaknesses. "
            + "After gathering the agent reviews, synthesize them and output ONLY the final line: \"FINAL DECISION: <Accept/Reject>\". "
            + "The paper title is: "
            + title
            + "\n\n"
        )

        # Reserve headroom so the initial call + a few "continue" rounds still fit.
        prompt_head_tokens_est = _estimate_tokens_from_chars(prompt_head, chars_per_token=chars_per_token)
        allowed_paper_tokens_est = max(
            0,
            max_input_tokens - prompt_head_tokens_est - reserve_history_tokens,
        )
        paper_text_truncated, did_truncate = truncate_text_to_token_budget(
            paper_text,
            token_budget=allowed_paper_tokens_est,
            chars_per_token=chars_per_token,
        )

        prompt = prompt_head + paper_text_truncated
        print(f"[{idx+1}/{len(df)}] Paper ID: {paper_id}")
        print(f"Title: {title}")
        print(f"Prompt: {prompt[:200]}...")
        print("→ Asking the agent to review the paper...")

        token_limit_error = None
        did_truncate_this_paper = did_truncate
        truncated_paper_chars = len(paper_text_truncated)

        try:
            start = time.time()
            resp, history = client.predict(
                {"text": prompt, "files": []},
                None,
                api_name="/chat",
            )

            content = get_last_assistant_content(history)
            print(content)
            rounds = 0

            lower_content = (content or "").lower()
            terminal_failure_markers = [
                "api key",
                "budget and api key issues",
                "not enough resources",
                "no enough resources",
                "connection error",
                "error asking agent",
                "unable to complete",
                "unable to proceed",
                "will have to stop here",
            ]
            def extract_accept_reject(text: str):
                """
                Extract Accept/Reject only from explicit verdict labels.
                Do not fall back to standalone Accept/Reject tokens, since those
                appear in reviewer sections and can cause misalignment.
                """
                if not text or not isinstance(text, str):
                    return None

                import re

                # Most specific: FINAL DECISION: Accept|Reject
                # Important: take the *last* occurrence, because some outputs include
                # multiple reviewer decisions before the final combined decision.
                final_matches = re.findall(
                    r"FINAL\s*DECISION\s*[:\-]\s*(Accept|Reject)",
                    text,
                    flags=re.IGNORECASE,
                )
                if final_matches:
                    return final_matches[-1].capitalize()

                # Common alternate: Recommendation: Accept|Reject
                rec_matches = re.findall(
                    r"Recommendation\s*[:\-]\s*(Accept|Reject)",
                    text,
                    flags=re.IGNORECASE,
                )
                if rec_matches:
                    return rec_matches[-1].capitalize()

                return None

            predicted_decision = extract_accept_reject(content)
            has_final = predicted_decision is not None

            def normalize_decision_label(s: str):
                """
                Normalize ground-truth formats like:
                - "Accept"
                - "Accept: poster"
                - "Reject: notable-top-5%"
                down to just "Accept" / "Reject".
                """
                if s is None:
                    return None
                import re
                t = str(s).strip()
                if not t:
                    return None
                if re.search(r"^\s*accept\b", t, flags=re.IGNORECASE):
                    return "Accept"
                if re.search(r"^\s*reject\b", t, flags=re.IGNORECASE):
                    return "Reject"
                # Also handle cases like "Accept: ..." not at start (defensive)
                if re.search(r"\baccept\s*:", t, flags=re.IGNORECASE):
                    return "Accept"
                if re.search(r"\breject\s*:", t, flags=re.IGNORECASE):
                    return "Reject"
                return None

            while not has_final and rounds < max_continue_rounds:
                # If the CEO is already reporting a terminal failure, "continue" just burns time.
                if any(m in lower_content for m in terminal_failure_markers):
                    print("  Detected terminal failure message (budget/api/connectivity). Not continuing.")
                    break
                sleep(continue_sleep_seconds)
                rounds += 1
                print(f"…no final verdict yet ({rounds}/{max_continue_rounds}), asking the agent to continue")
                try:
                    resp, history = client.predict(
                        {"text": "Return ONLY the final verdict. Reply with exactly one line: FINAL DECISION: Accept or FINAL DECISION: Reject.", "files": []},
                        history,
                        api_name="/chat",
                    )
                except Exception as e:
                    msg = str(e)
                    if "input token count" in msg and "exceeds" in msg and "1048575" in msg:
                        token_limit_error = msg
                        print("  Token limit exceeded; not continuing further.")
                        break
                    # Unknown failure: save partial and stop.
                    token_limit_error = msg
                    print(f"  Error on continue: {msg}")
                    break
                content = get_last_assistant_content(history)
                print(content)
                lower_content = (content or "").lower()
                predicted_decision = extract_accept_reject(content)
                has_final = predicted_decision is not None
                sleep(continue_sleep_seconds)
            elapsed = time.time() - start

            # Ensure we have a decision if the final message contains Accept/Reject anywhere.
            if predicted_decision is None:
                predicted_decision = extract_accept_reject(content)

            ground_truth = row["Decision"]
            gt_label = normalize_decision_label(ground_truth)
            pred_label = normalize_decision_label(predicted_decision)
            is_correct = (
                pred_label is not None
                and gt_label is not None
                and pred_label == gt_label
            )

            if not has_final:
                print(
                    f"  Warning: no Accept/Reject decision after {rounds} continue rounds; saving partial result."
                )

            def detect_tool_calls(agent_history):
                """
                Best-effort evidence detection:
                - `role: function_call` / `role: tool` in history indicates a tool was invoked.
                - Also scan for known tool names in content/metadata.
                """
                if not isinstance(agent_history, list):
                    return False
                tool_tokens = ["AskAgent", "AskMultipleAgents", "GetAgents", "compute_semantic_metrics"]
                for m in agent_history:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    if role in ("function_call", "tool"):
                        return True
                    for k in ("content",):
                        v = m.get(k)
                        if isinstance(v, str):
                            for t in tool_tokens:
                                if t in v:
                                    return True
                    md = m.get("metadata")
                    if isinstance(md, dict):
                        for v in md.values():
                            if isinstance(v, str):
                                for t in tool_tokens:
                                    if t in v:
                                        return True
                return False

            def detect_semantic_metrics(agent_history):
                if not isinstance(agent_history, list):
                    return False
                semantic_tokens = ["compute_semantic_metrics", "semantic_entropy", "semantic_density"]
                for m in agent_history:
                    if not isinstance(m, dict):
                        continue
                    for k in ("content",):
                        v = m.get(k)
                        if isinstance(v, str):
                            if any(tok in v for tok in semantic_tokens):
                                return True
                    md = m.get("metadata")
                    if isinstance(md, dict):
                        for v in md.values():
                            if isinstance(v, str):
                                if any(tok in v for tok in semantic_tokens):
                                    return True
                return False

            tool_calls_detected = detect_tool_calls(history)
            semantic_metrics_called = detect_semantic_metrics(history)

            # Store a short, normalized decision line for easier auditing.
            final_decision_line = None
            if content and isinstance(content, str):
                import re
                final_matches = re.findall(
                    r"FINAL\s*DECISION\s*:\s*(Accept|Reject)",
                    content,
                    flags=re.IGNORECASE,
                )
                if final_matches:
                    final_decision_line = f"FINAL DECISION: {final_matches[-1].capitalize()}"
                else:
                    rec_matches = re.findall(
                        r"Recommendation\s*[:\-]\s*(Accept|Reject)",
                        content,
                        flags=re.IGNORECASE,
                    )
                    if rec_matches:
                        final_decision_line = f"Recommendation: {rec_matches[-1].capitalize()}"
                    elif predicted_decision is not None:
                        final_decision_line = predicted_decision

            result = {
                "paper_id": paper_id,
                "prompt_snippet": prompt[:200],
                "ground_truth": ground_truth,
                "gt_label": gt_label,
                "predicted_decision": predicted_decision,
                "pred_label": pred_label,
                "is_correct": is_correct,
                "response_time": elapsed,
                "has_final_decision": has_final,
                "agents_called": tool_calls_detected,
                "semantic_metrics_called": semantic_metrics_called,
                "paper_truncated": did_truncate_this_paper,
                "paper_truncated_chars": truncated_paper_chars,
                "token_limit_error": token_limit_error is not None,
                "final_decision_line": final_decision_line,
            }

            # write immediately
            with open(out_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f" → {elapsed:.2f}s, final_decision={predicted_decision!r}")
            results.append(result)

            # small delay
            time.sleep(1)
        except Exception as e:
            print(f"  Error on {paper_id}: {e}")

    print(f"\nDone. Results written to {out_path}")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run paper-review benchmark.")
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of papers to review (default: all papers in the CSV)."
    )
    parser.add_argument(
        "--offset", "-o",
        type=int,
        default=0,
        help="Zero-based row index to start from (default: 0)."
    )
    parser.add_argument(
        "--max-continue-rounds",
        type=int,
        default=10,
        help="Max continue prompts while waiting for FINAL DECISION (default: 10)."
    )
    args = parser.parse_args()

    benchmark_paper_reviews(
        csv_path="data/ICLR_2023.csv",
        num_samples=args.num_samples,
        offset=args.offset,
        output_dir="results",
        max_continue_rounds=args.max_continue_rounds,
    )
