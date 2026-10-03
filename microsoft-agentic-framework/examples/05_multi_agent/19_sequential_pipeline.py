"""
Example 19 - Multi-Agent: Sequential Pipeline  (Level: Multi-Agent Systems)
==========================================================================

GOAL
    Build an assembly line of agents where each one builds on the previous
    agent's work: Researcher -> Writer -> Editor.

WHAT YOU WILL LEARN
    * ``SequentialBuilder`` - the simplest multi-agent orchestration.
    * Every agent sees the FULL conversation so far (user prompt + previous
      agents' messages), so later agents can refine earlier output.
    * The last agent's answer is the workflow ``output``; earlier agents can be
      surfaced as ``intermediate`` events with ``intermediate_output_from``.

WHEN TO USE
    Fixed, linear processes: draft -> review -> polish, extract -> transform
    -> summarise, translate -> localise -> proofread.

RUN
    python examples/05_multi_agent/19_sequential_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent  # noqa: E402
from agent_framework.orchestrations import SequentialBuilder  # noqa: E402

from common import get_chat_client, print_header, print_section, print_workflow_events  # noqa: E402


async def main() -> None:
    print_header(
        "19 - Sequential Pipeline",
        "Researcher -> Writer -> Editor. Each agent refines the previous agent's output.",
    )
    client = get_chat_client()

    researcher = Agent(
        client=client,
        name="Researcher",
        instructions="Collect 5 key facts (with numbers where possible) about the user's topic. Bullet points only.",
    )
    writer = Agent(
        client=client,
        name="Writer",
        instructions="Turn the researcher's facts into a 150-word LinkedIn post with a hook in the first line.",
    )
    editor = Agent(
        client=client,
        name="Editor",
        instructions=(
            "Edit the writer's post: fix grammar, tighten wording, add 3 relevant hashtags. "
            "Return ONLY the final post."
        ),
    )

    # Order of participants = execution order.
    workflow = SequentialBuilder(
        participants=[researcher, writer, editor],
        # By default only the LAST agent's answer is a workflow "output".
        # Listing the other agents here also surfaces their turns as
        # "intermediate" events, so we can watch the pipeline step by step.
        intermediate_output_from=[researcher, writer],
    ).build()

    print_section("Running the pipeline")
    # Non-streaming run: returns a WorkflowRunResult (a list of events).
    result = await workflow.run("The impact of AI agents on software development productivity")

    # Print each agent's turn (intermediate events) and the final output.
    print_workflow_events(result)


if __name__ == "__main__":
    asyncio.run(main())
