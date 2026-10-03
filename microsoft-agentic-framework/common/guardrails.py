"""
common/guardrails.py
====================

Re-usable, *deterministic* (rule-based) guardrails used by the guardrail
examples (13-15) and the capstone (25).

What is a guardrail?
--------------------
A guardrail is a check that sits between the user, the model and the tools and
enforces a policy, for example:

* INPUT guardrails  - block prompt-injection / jailbreak attempts, strip PII
                      before it is sent to the model, refuse banned topics.
* OUTPUT guardrails - stop the model from leaking PII or secrets, enforce
                      length/format, replace unsafe answers.
* TOOL guardrails   - only allow certain tools, validate arguments, rate-limit.

In the Microsoft Agent Framework guardrails are implemented as *middleware*
(see examples 07 and 08), so they can be attached to any agent without
changing the agent's code. This module provides:

1. Pure detection functions (easy to unit-test): ``find_pii``, ``redact_pii``,
   ``detect_prompt_injection`` and ``find_blocked_topics``.
2. Two ready-made agent middlewares built on top of them:
   ``InputGuardrailMiddleware`` and ``OutputGuardrailMiddleware``.

Rule-based checks are fast, free and predictable, but they can be bypassed by
creative phrasing. Example 16 shows how to complement them with an
LLM-based safety classifier. Production systems usually layer both, plus a
managed service such as Azure AI Content Safety / Prompt Shields.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable

from agent_framework import AgentContext, AgentMiddleware, AgentResponse, Message

# ---------------------------------------------------------------------------
# 1. PII (Personally Identifiable Information) detection
# ---------------------------------------------------------------------------
# Each entry: label -> compiled regex. These are intentionally simple patterns
# for teaching purposes; tune them for your country/data formats.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d{1,3}?[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
    "API_KEY": re.compile(r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE),
}


def find_pii(text: str) -> list[str]:
    """Return the sorted list of PII *types* found in ``text`` (e.g. ['EMAIL'])."""
    return sorted(label for label, pattern in PII_PATTERNS.items() if pattern.search(text))


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace every PII match with a placeholder like ``[REDACTED_EMAIL]``.

    Returns:
        (redacted_text, list_of_pii_types_found)
    """
    found: list[str] = []
    for label, pattern in PII_PATTERNS.items():
        text, count = pattern.subn(f"[REDACTED_{label}]", text)
        if count:
            found.append(label)
    return text, sorted(found)


# ---------------------------------------------------------------------------
# 2. Prompt-injection / jailbreak heuristics
# ---------------------------------------------------------------------------
PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)",
        r"disregard (your|the) (system )?(prompt|instructions|rules)",
        r"you are now (dan|in developer mode|unrestricted)",
        r"(reveal|print|show|repeat) (your|the) (system prompt|hidden instructions|instructions)",
        r"pretend (you have|there are) no (rules|restrictions|guidelines)",
        r"\bjailbreak\b",
    )
]


def detect_prompt_injection(text: str) -> list[str]:
    """Return the patterns that matched (empty list = looks clean)."""
    return [p.pattern for p in PROMPT_INJECTION_PATTERNS if p.search(text)]


# ---------------------------------------------------------------------------
# 3. Topic restrictions (simple keyword deny-list)
# ---------------------------------------------------------------------------
DEFAULT_BLOCKED_TOPICS: dict[str, tuple[str, ...]] = {
    "weapons": ("build a bomb", "make a weapon", "explosive recipe"),
    "malware": ("write ransomware", "keylogger code", "create a virus"),
    "self-harm": ("how to hurt myself", "ways to self-harm"),
}


def find_blocked_topics(text: str, blocked: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Return the names of blocked topics mentioned in ``text``."""
    lowered = text.lower()
    rules = blocked or DEFAULT_BLOCKED_TOPICS
    return [topic for topic, phrases in rules.items() if any(phrase in lowered for phrase in phrases)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def refusal(text: str) -> AgentResponse:
    """Build an AgentResponse that replaces the model's answer with ``text``."""
    return AgentResponse(messages=[Message(role="assistant", contents=[text])])


def _user_indices(messages: Iterable[Message]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.role == "user"]


# ---------------------------------------------------------------------------
# 4. Ready-made middleware
# ---------------------------------------------------------------------------
class InputGuardrailMiddleware(AgentMiddleware):
    """Checks the USER input BEFORE the agent/model sees it.

    Policy:
        * prompt injection  -> block the run and return a refusal
        * blocked topic     -> block the run and return a refusal
        * PII               -> redact it in place, then continue

    Note: for simplicity these guardrails are designed for non-streaming runs.
    """

    def __init__(self, *, redact: bool = True, blocked_topics: dict[str, tuple[str, ...]] | None = None) -> None:
        self.redact = redact
        self.blocked_topics = blocked_topics
        self.events: list[str] = []  # audit trail of what the guardrail did

    async def process(self, context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
        for index in _user_indices(context.messages):
            text = context.messages[index].text

            if injections := detect_prompt_injection(text):
                self.events.append(f"BLOCKED prompt injection: {injections[0]}")
                context.result = refusal("I can't follow instructions that try to override my safety rules.")
                return  # do NOT call call_next() -> the model is never invoked

            if topics := find_blocked_topics(text, self.blocked_topics):
                self.events.append(f"BLOCKED topic: {', '.join(topics)}")
                context.result = refusal(f"Sorry, I can't help with that topic ({', '.join(topics)}).")
                return

            if self.redact:
                redacted, pii = redact_pii(text)
                if pii:
                    self.events.append(f"REDACTED input PII: {', '.join(pii)}")
                    # Replace the message so the model only sees the redacted text.
                    context.messages[index] = Message(role="user", contents=[redacted])

        await call_next()


class OutputGuardrailMiddleware(AgentMiddleware):
    """Checks the AGENT output AFTER the model answered.

    Policy:
        * PII in the answer          -> redact it
        * blocked topic in the answer -> replace the whole answer
        * answer longer than limit    -> truncate it
    """

    def __init__(self, *, max_chars: int = 2_000) -> None:
        self.max_chars = max_chars
        self.events: list[str] = []

    async def process(self, context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()  # let the agent run first

        result = context.result
        if not isinstance(result, AgentResponse):
            return  # streaming result or nothing to check

        text = result.text
        if topics := find_blocked_topics(text):
            self.events.append(f"REPLACED unsafe output: {', '.join(topics)}")
            context.result = refusal("The generated answer was withheld because it violated the content policy.")
            return

        new_text, pii = redact_pii(text)
        if pii:
            self.events.append(f"REDACTED output PII: {', '.join(pii)}")
        if len(new_text) > self.max_chars:
            self.events.append(f"TRUNCATED output from {len(new_text)} to {self.max_chars} chars")
            new_text = new_text[: self.max_chars] + " ...[truncated]"
        if new_text != text:
            context.result = refusal(new_text)
