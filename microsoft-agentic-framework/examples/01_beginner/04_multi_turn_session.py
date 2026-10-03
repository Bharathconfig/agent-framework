"""
Example 04 - Multi-turn Conversation with Sessions  (Level: Beginner)
====================================================================

GOAL
    Have a real back-and-forth conversation in which the agent remembers what
    was said earlier.

WHAT YOU WILL LEARN
    * LLMs are stateless - every call is independent.
    * An ``AgentSession`` stores the conversation so follow-up questions work.
    * ``agent.create_session()`` + ``agent.run(..., session=session)``.
    * The difference between running *with* and *without* a session.

HOW IT WORKS
    When you pass a session, the framework automatically attaches an
    ``InMemoryHistoryProvider``. Before each run it loads previous messages
    from ``session.state`` and after each run it stores the new ones.

RUN
    python examples/01_beginner/04_multi_turn_session.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402


async def main() -> None:
    print_header(
        "04 - Multi-turn Session",
        "Shows that an agent only remembers previous turns when you give it a session.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="MemoryAgent",
        instructions="You are a concise assistant. Keep answers under two sentences.",
    )

    # ------------------------------------------------------------------
    # Part 1: WITHOUT a session - each call is a brand-new conversation.
    # ------------------------------------------------------------------
    print_section("Without a session (the agent forgets)")
    await agent.run("Hi! My name is Priya and I love astronomy.")
    response = await agent.run("What is my name and what do I love?")
    print_agent("User", "What is my name and what do I love?")
    print_agent(agent.name or "Agent", response.text)

    # ------------------------------------------------------------------
    # Part 2: WITH a session - the conversation history is kept.
    # ------------------------------------------------------------------
    print_section("With a session (the agent remembers)")
    session = agent.create_session()  # A fresh, empty conversation.

    turns = [
        "Hi! My name is Priya and I love astronomy.",
        "Recommend one planet I should observe tonight with a small telescope.",
        "What is my name and what hobby did I mention?",
    ]
    for user_text in turns:
        print_agent("User", user_text)
        # Passing the SAME session object on every call is what gives memory.
        response = await agent.run(user_text, session=session)
        print_agent(agent.name or "Agent", response.text)

    print(f"\nSession id: {session.session_id}")


if __name__ == "__main__":
    asyncio.run(main())
