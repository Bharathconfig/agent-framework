"""
Example 09 - Context Providers: Long-term Memory  (Level: Intermediate)
======================================================================

GOAL
    Give an agent a *memory* that remembers facts about the user and injects
    them into every future conversation - even brand-new sessions.

WHAT YOU WILL LEARN
    * ``ContextProvider`` is the framework's hook for "context engineering":
        - ``before_run``: add instructions / messages / tools before the LLM call.
        - ``after_run``:  inspect the request + response and update memory.
    * Provider-scoped ``state`` (stored inside the session) vs. provider
      instance attributes (shared across sessions = long-term memory).
    * Combining your provider with ``InMemoryHistoryProvider`` so you keep
      normal chat history too.

HOW THIS MEMORY WORKS
    After each run, a tiny "extractor" agent with structured output reads the
    user's message and returns any new personal facts. They are saved in a
    dictionary. Before each run, those facts are added to the instructions.

RUN
    python examples/02_intermediate/09_context_provider_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
from typing import Any  # noqa: E402

from agent_framework import (  # noqa: E402
    Agent,
    AgentSession,
    ContextProvider,
    InMemoryHistoryProvider,
    SessionContext,
    SupportsAgentRun,
)
from pydantic import BaseModel, Field  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402


class ExtractedFacts(BaseModel):
    """Schema for the extractor agent's answer."""

    facts: list[str] = Field(
        default_factory=list,
        description=(
            "Durable personal facts the user stated about themselves (name, job, preferences...). Empty if none."
        ),
    )


class UserMemoryProvider(ContextProvider):
    """Long-term user memory implemented as a context provider."""

    def __init__(self, extractor: Agent) -> None:
        # source_id identifies this provider (used for attribution / state keys).
        super().__init__(source_id="user_memory")
        self.extractor = extractor
        # Stored on the INSTANCE -> survives across sessions (long-term memory).
        # In production you'd persist this in a database keyed by user id.
        self.facts: list[str] = []

    async def before_run(
        self, *, agent: SupportsAgentRun, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        # Count turns in session-scoped state (resets for every new session).
        state["turns"] = state.get("turns", 0) + 1
        if self.facts:
            # Inject what we know about the user as extra system instructions.
            memory_block = "\n".join(f"- {fact}" for fact in self.facts)
            context.extend_instructions(self.source_id, f"Known facts about the user:\n{memory_block}")

    async def after_run(
        self, *, agent: SupportsAgentRun, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        # Look only at what the USER said in this turn.
        user_text = "\n".join(m.text for m in context.input_messages if m.role == "user")
        if not user_text.strip():
            return
        result = await self.extractor.run(user_text, options={"response_format": ExtractedFacts})
        extracted = result.value
        if extracted:
            for fact in extracted.facts:
                if fact not in self.facts:
                    self.facts.append(fact)
                    print(f"  [memory] learned: {fact}")


async def main() -> None:
    print_header(
        "09 - Context Provider Memory",
        "The agent learns facts in session 1 and still knows them in a brand-new session 2.",
    )
    client = get_chat_client()

    # A small helper agent whose only job is to extract facts.
    extractor = Agent(
        client=client,
        name="FactExtractor",
        instructions="Extract durable personal facts the user states about themselves. Ignore questions.",
    )
    memory = UserMemoryProvider(extractor)

    agent = Agent(
        client=client,
        name="PersonalAssistant",
        instructions="You are a helpful personal assistant. Personalise answers using known user facts.",
        # Because we register our own provider, we also add the history provider
        # explicitly (it is only auto-added when no providers are configured).
        context_providers=[InMemoryHistoryProvider(), memory],
    )

    print_section("Session 1")
    session1 = agent.create_session()
    for text in ["Hi, I'm Arjun. I'm a vegetarian and I work as a data scientist in Bengaluru.", "Thanks!"]:
        print_agent("User", text)
        print_agent(agent.name or "Agent", (await agent.run(text, session=session1)).text)

    print_section("Session 2 (fresh conversation, no chat history)")
    session2 = agent.create_session()
    question = "Suggest a dinner idea for me and a weekend activity near where I live."
    print_agent("User", question)
    print_agent(agent.name or "Agent", (await agent.run(question, session=session2)).text)

    print("\nEverything the memory knows:", memory.facts)


if __name__ == "__main__":
    asyncio.run(main())
