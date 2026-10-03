"""
Example 06 - Personas, Options and Per-run Overrides  (Level: Beginner)
======================================================================

GOAL
    Build several specialised agents from ONE chat client and learn how to
    control their behaviour with instructions and run options.

WHAT YOU WILL LEARN
    * One client can power many agents - each with its own persona.
    * ``default_options`` (agent-level) vs ``options=`` (single run) - the
      per-run value wins.
    * ``tool_choice`` to force/forbid tool usage.
    * Running agents in parallel with ``asyncio.gather`` for speed.

RUN
    python examples/01_beginner/06_personas_and_options.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent, tool  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


async def main() -> None:
    print_header(
        "06 - Personas and Options",
        "Three personas answer the same question in parallel, then we override options for one run.",
    )

    # One client, shared by all agents (re-uses the HTTP connection pool).
    client = get_chat_client()

    # Each persona is just a different set of instructions.
    personas = {
        "Teacher": "You explain things to a 10-year-old using a simple analogy. Max 3 sentences.",
        "Engineer": "You are a senior software engineer. Be precise and technical. Max 3 sentences.",
        "Poet": "You answer only with a 4-line rhyming poem.",
    }
    agents = [
        Agent(client=client, name=name, instructions=instructions, default_options={"temperature": 0.7})
        for name, instructions in personas.items()
    ]

    question = "What is the internet?"
    print_agent("User", question)

    # asyncio.gather runs all three requests concurrently -> ~3x faster than sequential.
    responses = await asyncio.gather(*(agent.run(question) for agent in agents))
    for agent, response in zip(agents, responses, strict=True):
        print_agent(agent.name or "Agent", response.text)

    # ------------------------------------------------------------------
    # Per-run overrides
    # ------------------------------------------------------------------
    print_section("Per-run option overrides")
    editor = Agent(
        client=client,
        name="Editor",
        instructions="You are an editor. Use the word_count tool when asked about length.",
        tools=[word_count],
        default_options={"max_tokens": 300},
    )

    text = "Agents combine language models with tools, memory and planning."

    # tool_choice="required" FORCES the model to call a tool at least once.
    forced = await editor.run(f"How long is this text? '{text}'", options={"tool_choice": "required"})
    print_agent("Editor (tool_choice=required)", forced.text)

    # tool_choice="none" FORBIDS tool calls for this run only, and we also
    # lower max_tokens just for this call.
    no_tools = await editor.run(
        f"Rewrite this text to sound more exciting: '{text}'",
        options={"tool_choice": "none", "max_tokens": 80},
    )
    print_agent("Editor (tool_choice=none)", no_tools.text)


if __name__ == "__main__":
    asyncio.run(main())
