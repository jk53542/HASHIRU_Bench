"""
Shared CEO delegation prefix for HASHIRU benchmarks.

HASHIRU activates worker enforcement when the user message contains
``IMPORTANT CEO INSTRUCTIONS`` and ``You MUST use agents`` (see HASHIRU_modified manager).

After a limited number of soft nudges (``HASHIRU_MANDATE_ASK_AGENT_MAX_NUDGES``, default 4),
the CEO may still answer without workers unless you set
``HASHIRU_STRICT_WORKER_MANDATE=1`` in the HASHIRU server environment so non-compliant
assistant text is discarded and the model is re-prompted (see ``manager.py``).

For benchmarks where you cannot tolerate CEO-only answers (e.g. StrategyQA ablations),
set ``HASHIRU_STRICT_WORKER_MANDATE=1`` on the HASHIRU process. Optional:
``HASHIRU_STRICT_WORKER_MANDATE_MAX_LOOPS`` (default 24) and ``HASHIRU_REQUIRE_ASK_AGENT=1``
(global ask-agent enforcement segment in ``manager.py``).

Import from sibling modules with: ``from benchmark_ceo_mandate import ...``
(Run benchmarks from this ``bench/`` directory so the import resolves.)
"""

# Core text — use for tau2 / jailbreak / any benchmark (enforcement trigger)
CEO_FORCE_AGENTS_PREFIX_CORE = (
    "IMPORTANT CEO INSTRUCTIONS:\n"
    "- You MUST use agents to solve this. Do NOT answer directly.\n"
    "- Do NOT rely only on tools/web search; delegate reasoning to one or more agents.\n"
    "- Reuse existing agents when possible; create a new agent only if a genuinely new specialty is required.\n"
    "- If the task is complex/multi-faceted, you may ask multiple agents and then synthesize.\n"
)

# StrategyQA: final answer must be JSON, and delegation must happen before the final JSON.
CEO_FORCE_AGENTS_PREFIX_STRATEGYQA = (
    CEO_FORCE_AGENTS_PREFIX_CORE
    + "- You MUST call AskAgent or AskMultipleAgents at least once before you output any final "
    "{\"answer\":\"...\"} JSON.\n"
    + "- Do not answer from the orchestrator model alone; base the final yes/no on worker output.\n"
    + "- After agents respond, provide ONLY the final JSON answer (no extra prose).\n"
)

# Default prefix for benchmarks that only need delegation + a free-form final answer format.
CEO_FORCE_AGENTS_PREFIX = (
    CEO_FORCE_AGENTS_PREFIX_CORE
    + "- After agents respond, provide ONLY the final JSON answer.\n"
)

# TruthfulQA, Jailbreak, etc.: task body below states the exact final format
CEO_FORCE_AGENTS_PREFIX_DELEGATE_THEN_TASK = (
    CEO_FORCE_AGENTS_PREFIX_CORE
    + "- After agents respond, follow the task instructions below for your user-facing final answer.\n"
)

# Multi-hop / short-answer QA (HotpotQA, MuSiQue): final answer is a short span.
CEO_FORCE_AGENTS_PREFIX_SHORT_ANSWER = (
    CEO_FORCE_AGENTS_PREFIX_CORE
    + "- You MUST call AskAgent or AskMultipleAgents at least once before the final answer.\n"
    + "- After agents respond, provide ONLY a final JSON answer: {\"answer\":\"<SHORT_ANSWER>\"}.\n"
    + "- The answer should be a concise span (a name, date, number, or short phrase). No extra prose.\n"
)

# Multiple-choice benchmarks (GPQA, MuSR, BBH): final answer is a single letter.
CEO_FORCE_AGENTS_PREFIX_MCQ = (
    CEO_FORCE_AGENTS_PREFIX_CORE
    + "- You MUST call AskAgent or AskMultipleAgents at least once before the final answer.\n"
    + "- After agents respond, provide ONLY a final JSON answer: {\"choice\":\"<LETTER>\"}.\n"
    + "- <LETTER> must be one of the option labels shown in the task (typically A, B, C, or D).\n"
)
