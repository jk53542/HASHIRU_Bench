import os
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from gradio_client import Client
from google import genai
from google.genai import types
from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_DELEGATE_THEN_TASK
from benchmark_trace_context import hashiru_trace_context_prefix

API_KEY = ""
random.seed(12345)

# Gradio HTTP timeout (seconds). MMLU-Pro few-shot prompts are huge; default client 30s breaks under load.
_GRADIO_TIMEOUT = float(os.environ.get("HASHIRU_BENCH_GRADIO_TIMEOUT", "3600"))
# Optional pause after each question (WSL memory / GPU recovery).
_INTER_Q_SLEEP = float(os.environ.get("HASHIRU_BENCH_MMLU_INTER_QUESTION_SLEEP", "0"))


def _make_gradio_client(url: str) -> Client:
    try:
        return Client(url, httpx_kwargs={"timeout": _GRADIO_TIMEOUT})
    except TypeError:
        return Client(url)


def _apply_mmlu_cot_limits(rows, max_examples: int, max_chars: int):
    """
    Slice and optionally truncate MMLU-Pro validation few-shot rows. The dataset ships one row per
    validation question per category; using all of them repeats enormous CoT in every benchmark call.
    """
    if not rows:
        return []
    out = [dict(r) for r in rows]
    if max_examples >= 0:
        out = out[:max_examples]
    if max_chars and max_chars > 0:
        for c in out:
            cc = c.get("cot_content") or ""
            if isinstance(cc, str) and len(cc) > max_chars:
                c["cot_content"] = cc[:max_chars] + "\n...[truncated for benchmark prompt size]"
    return out


def get_client():
    if args.model_name in ["hashiru"]:
        url = os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860/")
        client = _make_gradio_client(url)
        client.predict(
            modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
            api_name="/update_model"
        )
        return client
    elif args.model_name in ["flash2.0"]:
        client = genai.Client(api_key=API_KEY)
        return client


def _get_last_assistant_content(resp):
    """Best-effort extraction of final assistant text from Gradio history."""
    if isinstance(resp, tuple):
        resp = resp[0]
    if not isinstance(resp, list):
        return ""
    for turn in reversed(resp):
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        c = turn.get("content")
        if isinstance(c, str) and c:
            return c
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return str(out)
    return ""


def _is_tool_loop_guard_text(s: str) -> bool:
    t = (s or "").lower()
    return (
        "tool-loop guard" in t
        or "maximum tool rounds reached" in t
        or "without more tool calls" in t
    )


def _hashiru_mmlu_scoring_text(history) -> str:
    """
    Build text for MMLU letter extraction: all assistant strings + tool outputs, forward order,
    skipping the final tool-loop guard message (CEO often ends with that while worker had the MCQ).
    """
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return ""
    parts: list[str] = []
    for turn in history:
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        c = turn.get("content")
        if isinstance(c, str) and c.strip():
            if not _is_tool_loop_guard_text(c):
                parts.append(c.strip())
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out is not None:
            parts.append(str(out).strip())
    return "\n".join(p for p in parts if p)


def call_api(client, instruction, inputs, tries=0, trace_prefix="", prompt_body=""):
    start = time.time()
    if args.model_name in ["hashiru"]:
        if tries > 3:
            print("Error: too many tries")
            return ""
        # Reuse the client from evaluate(); creating a new Client every question re-fetches /config
        # (short default timeout) and duplicates the giant MMLU prompt cost on the wire.
        cli = client if client is not None else _make_gradio_client(
            os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860/")
        )
        message_text = (trace_prefix + (prompt_body or (instruction + inputs))).strip()
        response, history = cli.predict(
                    message={"text": message_text, "files": []},
                    api_name="/chat"
                )
        content = response.get("content", "") if isinstance(response, dict) else ""
        if isinstance(content, str) and "error" in content.lower():
            time.sleep(60)
            response = call_api(
                cli,
                instruction,
                inputs,
                tries + 1,
                trace_prefix=trace_prefix,
                prompt_body=prompt_body,
            )
            return response
            
        print("cost time", time.time() - start)
        blob = _hashiru_mmlu_scoring_text(history)
        last = _get_last_assistant_content(history) or (response.get("content", "") if isinstance(response, dict) else "") or ""
        if try_extract_mmlu_letter(blob):
            return blob
        if try_extract_mmlu_letter(last):
            return last
        return (blob + "\n" + last).strip() or last
    elif args.model_name in ["flash2.0"]:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=instruction + inputs,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    safety_settings=safety_settings,
                ),
            )
        except Exception as e:
            if tries > 3:
                print("Error: too many tries")
                return ""
            time.sleep(60)
            output = call_api(client, instruction, inputs, tries + 1)
            return output

        print("cost time", time.time() - start)
        return response.text


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def try_extract_mmlu_letter(text: str):
    """Return A–J if found, else None (no logging)."""
    if not text or not str(text).strip():
        return None
    t = str(text)
    m = re.search(r"answer\s+is\s*\(?([A-Ja-j])\)?", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:^|[\n.])\s*answer\s*:\s*\(?([A-Ja-j])\)?", t, re.MULTILINE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\boption\s*\(?([A-Ja-j])\)?\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def extract_answer(text, verbose=True):
    direct = try_extract_mmlu_letter(text)
    if direct:
        return direct
    if verbose:
        clip = (text or "")[:2500]
        print("1st answer extract failed\n" + clip)
    return extract_again(text)


def extract_again(text):
    direct = try_extract_mmlu_letter(text)
    if direct:
        return direct
    match = re.search(r'.*[aA]nswer:\s*([A-Ja-j])', text)
    if match:
        return match.group(1).upper()
    return extract_final(text)


def extract_final(text):
    direct = try_extract_mmlu_letter(text)
    if direct:
        return direct
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    category = single_question["category"]
    full_cot = cot_examples_dict[category]
    cot_examples = _apply_mmlu_cot_limits(
        full_cot, args.max_cot_examples, args.max_cot_chars
    )
    question = single_question["question"]
    options = single_question["options"]
    # prompt = (
    #         f"The following are multiple choice questions (with answers) about {category}"
    #         "You MUST NOT ask the user for clarifications. "
    #         "You MUST use tools/agents to help you. "
    #         "Deep-research and answer the question always and then output the answer in the format of "
    #         "\"The answer is (X)\" at the end.\n\n"
    #     )
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    task_body = (
        f"{CEO_FORCE_AGENTS_PREFIX_DELEGATE_THEN_TASK}\n"
        f"{prompt}{input_text}"
    )
    trace_prefix = hashiru_trace_context_prefix(
        benchmark_name="mmlu_pro",
        question_index=int(single_question.get("_bench_index", 0)),
        question_id=str(single_question.get("question_id", "")),
        bench_attempt=1,
        question_text=(question or "")[:2500],
    )
    try:
        response = call_api(
            client,
            prompt,
            input_text,
            trace_prefix=trace_prefix,
            prompt_body=task_body,
        )
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return pred, response, exist


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects):
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    available = list(test_df.keys())
    if not subjects:
        subjects = available
    else:
        # Validate and normalize: MMLU-Pro category names may differ (e.g. spaces vs underscores)
        normalized = {}
        for k in available:
            normalized[k.lower().replace(" ", "_")] = k
        resolved = []
        for s in subjects:
            key = s.strip().lower().replace(" ", "_")
            if key in normalized:
                resolved.append(normalized[key])
            elif s in test_df:
                resolved.append(s)
            else:
                print(f"Unknown subject '{s}'. Available subjects: {available}")
        subjects = resolved if resolved else available
    print("assigned subjects", subjects)
    for subject in subjects:
        full_dev = dev_df[subject]
        limited = _apply_mmlu_cot_limits(
            full_dev, args.max_cot_examples, args.max_cot_chars
        )
        print(
            f"[{subject}] MMLU-Pro few-shot: {len(limited)}/{len(full_dev)} validation examples "
            f"(max_cot_examples={args.max_cot_examples}, max_cot_chars={args.max_cot_chars})"
        )
        test_data = test_df[subject]
        if args.offset > 0 or args.num_samples is not None:
            start = max(0, int(args.offset))
            end = len(test_data) if args.num_samples is None else min(
                len(test_data), start + max(0, int(args.num_samples))
            )
            test_data = test_data[start:end]
            print(
                f"[{subject}] evaluating sliced range start={start}, end={end}, n={len(test_data)} "
                f"(subject_total={len(test_df[subject])})"
            )
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        for idx, each in enumerate(tqdm(test_data), start=1):
            each["_bench_index"] = idx
            label = each["answer"]
            category = subject
            pred, response, exist = single_request(client, each, dev_df, res)
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
                if _INTER_Q_SLEEP > 0:
                    time.sleep(_INTER_Q_SLEEP)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4",
                        choices=["hashiru", "flash2.0"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all",
                        help="Comma-separated subject names, or 'all'. Use --list_subjects to print valid names.")
    parser.add_argument("--list_subjects", action="store_true", help="Load dataset and print available subject names, then exit.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Maximum number of questions per selected subject (default: all).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start index within each selected subject (default: 0).")
    parser.add_argument(
        "--max_cot_examples",
        type=int,
        default=None,
        help="Max validation few-shot rows per subject (each includes full CoT). "
        "Default: HASHIRU_BENCH_MMLU_MAX_COT_EXAMPLES or 5. Use -1 for no cap (original behavior; very large prompts).",
    )
    parser.add_argument(
        "--max_cot_chars",
        type=int,
        default=None,
        help="Truncate each few-shot cot_content to this many characters (0=off). "
        "Default: HASHIRU_BENCH_MMLU_MAX_COT_CHARS or 2000.",
    )
    assigned_subjects = []
    args = parser.parse_args()

    if args.max_cot_examples is None:
        raw = os.environ.get("HASHIRU_BENCH_MMLU_MAX_COT_EXAMPLES", "5").strip()
        args.max_cot_examples = int(raw) if raw else 5
    if args.max_cot_chars is None:
        raw = os.environ.get("HASHIRU_BENCH_MMLU_MAX_COT_CHARS", "2000").strip()
        args.max_cot_chars = int(raw) if raw else 2000

    if args.list_subjects:
        test_df, _ = load_mmlu_pro()
        print("Available MMLU-Pro subjects:")
        for s in sorted(test_df.keys()):
            print(f"  {s}")
        raise SystemExit(0)

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = [x.strip() for x in args.assigned_subjects.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)