"""
Example 21 - Multi-Agent: Group Chat with a Moderator  (Level: MAS)
==================================================================

GOAL
    Let several agents DEBATE in a shared conversation while a moderator agent
    decides who speaks next and when the discussion is finished.

WHAT YOU WILL LEARN
    * ``GroupChatBuilder`` - centralised orchestration: all participants see
      one shared conversation.
    * Two ways to pick the next speaker:
        1. ``orchestrator_agent=`` an LLM "moderator" that returns a structured
           decision (next speaker / terminate / final message).  <- used here
        2. ``selection_func=`` your own Python function (e.g. round-robin).
    * Safety nets: ``max_rounds`` so the chat can never run forever.

GROUP CHAT vs HANDOFF
    Group chat = a moderator decides (centralised).
    Handoff    = agents pass control to each other themselves (example 22).

RUN
    python examples/05_multi_agent/21_group_chat_debate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent  # noqa: E402
from agent_framework.orchestrations import GroupChatBuilder  # noqa: E402

from common import get_chat_client, print_header, print_section, print_workflow_events  # noqa: E402


async def main() -> None:
    print_header(
        "21 - Group Chat Debate",
        "An Optimist, a Skeptic and a Pragmatist debate; a Moderator agent manages turns and ends the debate.",
    )
    client = get_chat_client()

    # 'description' is what the moderator reads when choosing the next speaker.
    optimist = Agent(
        client=client,
        name="Optimist",
        description="Argues for the benefits and opportunities.",
        instructions="You are optimistic. Make ONE concise argument (max 60 words) and respond to others.",
    )
    skeptic = Agent(
        client=client,
        name="Skeptic",
        description="Points out risks, costs and flaws.",
        instructions="You are skeptical. Make ONE concise counter-argument (max 60 words) and respond to others.",
    )
    pragmatist = Agent(
        client=client,
        name="Pragmatist",
        description="Finds a practical middle ground and concrete actions.",
        instructions="You are pragmatic. Propose ONE balanced, actionable compromise (max 60 words).",
    )

    moderator = Agent(
        client=client,
        name="Moderator",
        instructions=(
            "You moderate a debate. Give each participant at least one turn, prefer the participant who "
            "has spoken least, and terminate once a practical compromise has been proposed and discussed. "
            "When terminating, write a 2-sentence summary as the final message."
        ),
    )

    workflow = GroupChatBuilder(
        participants=[optimist, skeptic, pragmatist],
        orchestrator_agent=moderator,  # LLM decides who speaks next and when to stop
        max_rounds=6,  # hard upper bound on the number of turns
        # Surface every participant's turn as an "intermediate" event so we can
        # print the whole debate (the final "output" is the moderator's summary).
        intermediate_output_from="all_other",
    ).build()

    topic = "Should companies allow AI agents to merge code to production without human review?"
    print_section("Debate")
    result = await workflow.run(topic)
    print_workflow_events(result)


if __name__ == "__main__":
    asyncio.run(main())
