"""
HASHIRU agent for τ²-bench (tau2-bench): forwards agent turns to a Gradio /chat endpoint.

Requires: pip install tau2-bench gradio_client

Use from benchmarking_tau2.py by registering this agent and running tau2 with --agent hashiru_agent.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from gradio_client import Client

from benchmark_ceo_mandate import CEO_FORCE_AGENTS_PREFIX_CORE
from benchmark_trace_context import hashiru_trace_context_prefix

# Tau2 imports (optional until used)
try:
    from tau2.agent.base import LocalAgent, ValidAgentInputMessage, is_valid_agent_history_message
    from tau2.agent.llm_agent import LLMAgent, LLMAgentState
    from tau2.data_model.message import (
        AssistantMessage,
        Message,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.environment.tool import Tool
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    ToolCall = None  # type: ignore


def _gradio_url_from_env_or_args(llm_args: Optional[dict]) -> str:
    if llm_args and isinstance(llm_args.get("gradio_url"), str):
        return llm_args["gradio_url"].rstrip("/")
    return os.environ.get("HASHIRU_GRADIO_URL", "http://127.0.0.1:7860").rstrip("/")


def _message_to_text(msg: Any) -> str:
    if hasattr(msg, "content") and msg.content:
        return msg.content
    if hasattr(msg, "tool_messages"):
        parts = [_tool_msg_line(m) for m in msg.tool_messages]
        return "\n".join(parts)
    if hasattr(msg, "role") and getattr(msg, "content", None):
        return msg.content or ""
    return ""


def _tool_msg_line(t: Any) -> str:
    content = getattr(t, "content", None) or ""
    err = getattr(t, "error", False)
    return f"[Tool result (error={err})] {content}" if content else "[Tool result]"


# Instruction prepended to EVERY message so the CEO sees it even if history is ignored or wrong format.
# Emphasize: reuse existing agents, create at most ONE agent per turn, then AskAgent once (avoids create-loop).
# ENV_CALL must use ONLY the database/simulation Tools listed in System (e.g. find_user_id_by_name_zip, get_order_details,
# get_product_details, exchange_delivered_order_items, modify_pending_order_items, return_delivered_order_items).
# Do NOT use ProductSearch, ReturnTool, GoogleSearchTool or any other internal name in ENV_CALL—only the listed Tools run against the simulation.
_MANDATORY_PREFIX = (
    "[RULES]\n"
    "1) Delegate via agents: First call GetAgents. If a suitable agent exists, use AskAgent to send this request to it. "
    "Only if none exists, create exactly ONE agent with AgentCreator (use one base model only), then use AskAgent once. "
    "Do NOT create the same agent repeatedly or try multiple models in a loop. Do NOT reply with only text.\n"
    "2) For database/simulation actions output one line: ENV_CALL: [{\"name\": \"<tool>\", \"arguments\": {...}}] "
    "using ONLY the Tools listed in [System] (e.g. find_user_id_by_name_zip, get_order_details, get_product_details, exchange_delivered_order_items, modify_pending_order_items, return_delivered_order_items). "
    "Do NOT use ProductSearch, ReturnTool, or other internal names in ENV_CALL.\n"
    "---\n\n"
)


def _tau2_messages_to_gradio_history(messages: List[Message]) -> List[dict]:
    """Convert tau2 message list to Gradio chat history: list of {role, content}."""
    out = []
    for m in messages:
        if isinstance(m, UserMessage):
            if m.has_text_content():
                out.append({"role": "user", "content": m.content or ""})
            elif m.is_tool_call():
                out.append({"role": "user", "content": "[User made a tool call]. (Handled by environment.)"})
        elif isinstance(m, AssistantMessage):
            if m.has_text_content():
                out.append({"role": "assistant", "content": m.content or ""})
            elif m.is_tool_call():
                out.append({"role": "assistant", "content": "[Agent made tool call]. (Handled by environment.)"})
        elif isinstance(m, ToolMessage):
            out.append({"role": "user", "content": _tool_msg_line(m)})
    return out


def _gradio_history_to_tuples(history: List[dict]) -> List[tuple]:
    """Convert list of {role, content} to Gradio legacy tuple format [(user, bot), ...]."""
    pairs = []
    i = 0
    while i < len(history):
        msg = history[i]
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "user":
            bot_content = ""
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                bot_content = history[i + 1].get("content") or ""
                i += 1
            pairs.append((content, bot_content))
        elif role == "assistant":
            if pairs:
                pairs[-1] = (pairs[-1][0], content)
            else:
                pairs.append(("", content))
        i += 1
    return pairs


def _get_last_assistant_content(history: Any) -> str:
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return ""
    # Gradio may return list of (user, bot) tuples instead of list of dicts.
    if history and isinstance(history[-1], (list, tuple)) and len(history[-1]) >= 2:
        last = history[-1]
        if isinstance(last, (list, tuple)):
            return (last[1] or "") if len(last) > 1 else ""
    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "assistant":
            continue
        if turn.get("content"):
            return turn["content"]
        fr = turn.get("function_response", {}) or {}
        out = (fr.get("result") or {}).get("output")
        if out:
            return out
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts") or []
            if parts and parts[0].get("text"):
                return parts[0]["text"]
    return ""


def _get_all_assistant_content(history: Any) -> str:
    """Concatenate all assistant message content from history. Used to find ENV_CALL anywhere in the turn."""
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return ""
    parts = []
    for turn in history:
        content = ""
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            # (user_msg, bot_msg) format
            content = (turn[1] or "") if len(turn) > 1 else ""
        elif isinstance(turn, dict) and turn.get("role") == "assistant":
            content = turn.get("content") or ""
            if not content:
                fr = turn.get("function_response", {}) or {}
                content = (fr.get("result") or {}).get("output") or ""
        if content and isinstance(content, str):
            parts.append(content)
    return "\n\n".join(parts)


def _format_env_tools(tools: List[Any]) -> str:
    """Format tau2 environment tool names for the system prompt."""
    if not tools:
        return "(none)"
    names = sorted([getattr(t, "name", str(t)) for t in tools])
    return ", ".join(names)


def _normalize_tool_arguments(name: str, args: dict) -> dict:
    """Normalize arguments to match tau2 retail (and similar) tool signatures."""
    args = dict(args)
    # find_user_id_by_name_zip expects first_name, last_name, zip (not name, zip_code)
    if name == "find_user_id_by_name_zip":
        if "name" in args and ("first_name" not in args or "last_name" not in args):
            full = str(args.pop("name", "")).strip()
            if full:
                parts = full.split(None, 1)
                args["first_name"] = parts[0]
                args["last_name"] = parts[1] if len(parts) > 1 else ""
        if "zip_code" in args and "zip" not in args:
            args["zip"] = args.pop("zip_code")
    # order_id in retail must have leading '#' (e.g. #W2378156)
    if "order_id" in args and isinstance(args["order_id"], str):
        oid = args["order_id"].strip()
        if oid and not oid.startswith("#"):
            args["order_id"] = "#" + oid
    return args


def _parse_env_calls(text: str, valid_tool_names: set[str]) -> List[Any]:
    """
    Parse ENV_CALL: [...] or ENV_CALL: {...} from the assistant text. Returns list of ToolCall if TAU2_AVAILABLE.
    valid_tool_names: set of allowed tool names (from self.tools).
    Uses bracket matching so nested JSON in arguments is handled.
    Normalizes arguments for tau2 (e.g. find_user_id_by_name_zip: name/zip_code -> first_name, last_name, zip).
    """
    if not text or not valid_tool_names or not TAU2_AVAILABLE or ToolCall is None:
        return []
    m = re.search(r"ENV_CALL\s*:\s*", text)
    if not m:
        return []
    start = m.end()
    rest = text[start:].strip()
    if not rest or rest[0] not in "[{":
        return []
    open_b, close_b = ("[", "]") if rest[0] == "[" else ("{", "}")
    depth = 0
    end = -1
    for i, c in enumerate(rest):
        if c == open_b:
            depth += 1
        elif c == close_b:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end <= 0:
        return []
    json_str = rest[:end]
    try:
        raw = json.loads(json_str)
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name or name not in valid_tool_names:
                continue
            args = item.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                args = {}
            args = _normalize_tool_arguments(name, args)
            out.append(ToolCall(id=f"hashiru-{i}", name=name, arguments=args, requestor="assistant"))
        return out
    except (json.JSONDecodeError, TypeError):
        return []


if TAU2_AVAILABLE:

    class HashiruGradioAgent(LLMAgent):
        """
        Agent that sends each turn to a HASHIRU Gradio /chat endpoint (subclasses LLMAgent
        so tau2 run_task constructs it with tools, domain_policy, llm, llm_args).
        Uses llm_args["gradio_url"] or env HASHIRU_GRADIO_URL; llm is ignored.
        Prompts the CEO to use AgentCreator/AskAgent first, then output ENV_CALL: [...] for
        tau2 environment tools; parses ENV_CALL and returns tool_calls so tau2 can run them.
        """

        def __init__(
            self,
            tools: List[Tool],
            domain_policy: str,
            llm: Optional[str] = None,
            llm_args: Optional[dict] = None,
        ):
            super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args or {})
            self._gradio_url = _gradio_url_from_env_or_args(self.llm_args)
            self._client: Optional[Client] = None
            # Per-simulation Gradio /chat calls (one HASHIRU user turn per tau2 agent step).
            self._trace_gradio_turn: int = 0

        @property
        def client(self) -> Client:
            if self._client is None:
                self._client = Client(self._gradio_url + "/")
                try:
                    self._client.predict(
                        modeIndexes=[
                            "ENABLE_AGENT_CREATION",
                            "ENABLE_LOCAL_AGENTS",
                            "ENABLE_CLOUD_AGENTS",
                            "ENABLE_TOOL_CREATION",
                            "ENABLE_TOOL_INVOCATION",
                            "ENABLE_RESOURCE_BUDGET",
                            "ENABLE_ECONOMY_BUDGET",
                        ],
                        api_name="/update_model",
                    )
                except Exception:
                    pass  # /update_model may not exist on all Gradio apps
            return self._client

        def get_init_state(
            self,
            message_history: Optional[list[Message]] = None,
        ) -> LLMAgentState:
            if message_history is None:
                message_history = []
            assert all(
                is_valid_agent_history_message(m) for m in message_history
            ), "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
            system_content = (
                "You are a customer service agent. Follow the policy and use tools when needed. "
                "Reply with either a message to the user or with tool calls, not both.\n\n"
                "Policy:\n" + self.domain_policy
            )
            return LLMAgentState(
                system_messages=[SystemMessage(role="system", content=system_content)],
                messages=list(message_history),
            )

        def generate_next_message(
            self,
            message: ValidAgentInputMessage,
            state: LLMAgentState,
        ) -> tuple[AssistantMessage, LLMAgentState]:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)

            # History for Gradio = all messages except the last (the one we are "replying" to).
            conv_history = _tau2_messages_to_gradio_history(state.messages[:-1])
            next_user_text = _message_to_text(state.messages[-1]) if state.messages else "(Continue.)"

            # Critical: prepend mandatory instruction to the CURRENT message so the CEO sees it every turn
            # even if history is ignored, wrong format, or truncated.
            message_text = (
                CEO_FORCE_AGENTS_PREFIX_CORE
                + "\n"
                + _MANDATORY_PREFIX
                + (next_user_text or "(Continue.)")
            ).strip()

            self._trace_gradio_turn += 1
            la = self.llm_args or {}
            tau_domain = la.get("tau2_domain")
            bench_name = la.get("hashiru_benchmark_name") or (
                f"tau2_{tau_domain}" if tau_domain else "tau2"
            )
            task_id = la.get("tau2_task_id")
            trial_idx = la.get("tau2_trial")
            if task_id is not None:
                qid = f"{task_id}_trial{trial_idx + 1}" if trial_idx is not None else str(task_id)
            else:
                qid = None
            trace_prefix = hashiru_trace_context_prefix(
                benchmark_name=bench_name,
                question_index=self._trace_gradio_turn,
                question_id=qid,
                bench_attempt=1,
            )
            message_text = (trace_prefix + message_text).strip()

            valid_tool_names = {getattr(t, "name", str(t)) for t in self.tools}
            env_tools_str = _format_env_tools(self.tools)
            system_block = (
                "[System]\n"
                "Delegate every request via an agent: call GetAgents first; if a suitable agent exists use AskAgent. "
                "If none exists, create exactly ONE agent with AgentCreator (one model only), then use AskAgent once. "
                "Never create the same agent multiple times or try different models in a loop. No text-only replies.\n\n"
                "Database/simulation Tools (use ONLY these names in ENV_CALL): " + env_tools_str + "\n"
                "When you or your agent need to look up or change customer/order data, output exactly one line: "
                "ENV_CALL: [{\"name\": \"<tool>\", \"arguments\": {...}}] with one of the Tools above and correct arguments "
                "(e.g. first_name, last_name, zip for find_user_id_by_name_zip; order_id for get_order_details; product_id for get_product_details). "
                "Do NOT use ProductSearch, ReturnTool, or any other name in ENV_CALL.\n"
                "Example: ENV_CALL: [{\"name\": \"find_user_id_by_name_zip\", \"arguments\": {\"first_name\": \"Yusuf\", \"last_name\": \"Rossi\", \"zip\": \"19122\"}}]\n"
                "Use exact argument names: first_name, last_name, zip (not name or zip_code); order_id with '#' prefix (e.g. #W2378156).\n\n"
                "Policy:\n" + (self.domain_policy or "(none)")
            )
            gradio_history_dicts = [{"role": "user", "content": system_block}] + conv_history

            # Some Gradio apps expect history as list of (user, bot) tuples; default is list of dicts.
            history_to_send = (
                _gradio_history_to_tuples(gradio_history_dicts)
                if os.environ.get("TAU2_GRADIO_HISTORY_TUPLES") == "1"
                else gradio_history_dicts
            )
            try:
                result = self.client.predict(
                    {"text": message_text, "files": []},
                    history_to_send,
                    api_name="/chat",
                )
            except Exception as e:
                return (
                    AssistantMessage(
                        role="assistant",
                        content=f"[Error calling Gradio agent: {e}]",
                        tool_calls=None,
                    ),
                    state,
                )

            if isinstance(result, tuple):
                _, new_history = result
            else:
                new_history = result if isinstance(result, list) else []

            text = _get_last_assistant_content(new_history)
            if not text and isinstance(result, tuple) and len(result) > 0:
                text = str(result[0]) if result[0] else ""

            # Parse ENV_CALL from the FULL history: CEO may put ENV_CALL in an earlier turn, not the last message.
            full_assistant_text = _get_all_assistant_content(new_history)
            parsed_calls = _parse_env_calls(full_assistant_text or text or "", valid_tool_names)
            if parsed_calls:
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=parsed_calls,
                )
            else:
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content=text or "(No response.)",
                    tool_calls=None,
                )
            state.messages.append(assistant_msg)
            return assistant_msg, state

else:
    HashiruGradioAgent = None  # type: ignore
