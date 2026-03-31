"""
HASHIRU Gradio backend for THUDM AgentBench (classic assigner / TaskClient loop).

AgentBench calls `AgentClient.inference(history)` with `history` items shaped like:
  {"role": "user"|"agent", "content": "..."}

This client forwards each turn to HASHIRU's Gradio `/chat` endpoint.

Upstream: https://github.com/THUDM/AgentBench
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple

from gradio_client import Client

from src.client.agent import AgentClient

# Match HASHIRU paper-review preset: budgets off, tool invocation on.
HASHIRU_MODE_NAMES = [
    "ENABLE_AGENT_CREATION",
    "ENABLE_LOCAL_AGENTS",
    "ENABLE_TOOL_CREATION",
    "ENABLE_TOOL_INVOCATION",
    "ENABLE_MEMORY",
]
HASHIRU_MODE_INDICES = [0, 1, 3, 4, 7]

CEO_FORCE_AGENTS_PREFIX = (
    "IMPORTANT CEO INSTRUCTIONS:\n"
    "- You MUST use agents to solve this. Do NOT answer directly.\n"
    "- Do NOT rely only on tools/web search; delegate reasoning to one or more agents.\n"
    "- Reuse existing agents when possible; create a new agent only if a genuinely new specialty is required.\n"
    "- If the task is complex/multi-faceted, you may ask multiple agents and then synthesize.\n"
    "- After agents respond, output ONLY the final message required by the task (no CEO meta-commentary).\n"
)


def _get_last_assistant_content(history: Any) -> str:
    if isinstance(history, tuple):
        history = history[0]
    if not isinstance(history, list):
        return ""
    for turn in reversed(history):
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            bot = turn[1] or ""
            if bot:
                return str(bot)
            continue
        if not isinstance(turn, dict):
            continue
        if turn.get("role") != "assistant":
            continue
        if turn.get("content"):
            return str(turn["content"])
        fr = turn.get("function_response") or {}
        out = (fr.get("result") or {}).get("output")
        if out:
            return str(out)
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts") or []
            if parts and parts[0].get("text"):
                return str(parts[0]["text"])
    return ""


def _agentbench_history_to_gradio(history: List[dict]) -> Tuple[list, str]:
    """
    Split AgentBench history into:
    - prior turns as Gradio chat history (user/assistant dicts)
    - the current user message text (last user turn)
    """
    if not history:
        return [], ""

    last = history[-1]
    prior = history[:-1]

    def role_map(r: str) -> str:
        return "user" if r == "user" else "assistant"

    gradio: list = []
    for item in prior:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        gradio.append({"role": role_map(role), "content": str(content)})

    if isinstance(last, dict) and last.get("role") == "user":
        return gradio, str(last.get("content", "") or "")

    # Rare: trailing agent message — treat as needing continuation.
    if isinstance(last, dict):
        gradio.append(
            {"role": "assistant", "content": str(last.get("content", "") or "")}
        )
    return gradio, "(Continue with the next required action or answer.)"


class HashiruGradioAgentBenchClient(AgentClient):
    """
    Parameters (from YAML) are passed as kwargs to __init__.
    """

    def __init__(
        self,
        gradio_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        base = (gradio_url or os.environ.get("HASHIRU_GRADIO_URL") or "http://127.0.0.1:7860").rstrip(
            "/"
        )
        self._gradio_url = base + "/"
        self._client: Client | None = None
        self._modes_configured = False

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(self._gradio_url)
        return self._client

    def _configure_modes(self) -> None:
        if self._modes_configured:
            return
        for modes in (HASHIRU_MODE_NAMES, HASHIRU_MODE_INDICES):
            try:
                self.client.predict(modeIndexes=modes, api_name="/update_model")
                self._modes_configured = True
                return
            except Exception:
                continue
        self._modes_configured = True  # avoid tight loop; /chat may still work

    def inference(self, history: List[dict]) -> str:
        self._configure_modes()
        gradio_hist, user_text = _agentbench_history_to_gradio(history)
        message_text = (CEO_FORCE_AGENTS_PREFIX + "\n\n" + user_text).strip()
        hist_arg = gradio_hist if gradio_hist else None
        _resp, ghist = self.client.predict(
            {"text": message_text, "files": []},
            hist_arg,
            api_name="/chat",
        )
        out = _get_last_assistant_content(ghist)
        return out if out else "(no response)"
