"""
Example 20 - Multi-Agent: Concurrent Experts (Fan-out / Fan-in)  (Level: MAS)
============================================================================

GOAL
    Ask several expert agents the SAME question IN PARALLEL, then merge their
    opinions into one recommendation.

WHAT YOU WILL LEARN
    * ``ConcurrentBuilder`` - fan-out the input to all participants at once,
      then fan-in (aggregate) their answers.
    * ``.with_aggregator(callback)`` - a custom async function that receives
      ``list[AgentExecutorResponse]`` and returns the workflow output. Here the
      aggregator itself calls a "Synthesizer" agent.
    * Parallelism: total time ~= slowest expert, not the sum of all experts.

WHEN TO USE
    Multi-perspective analysis, ensembles/voting, parallel research on
    independent sub-topics.

RUN
    python examples/05_multi_agent/20_concurrent_experts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import time  # noqa: E402

from agent_framework import Agent, AgentExecutorResponse  # noqa: E402
from agent_framework.orchestrations import ConcurrentBuilder  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402


async def main() -> None:
    print_header(
        "20 - Concurrent Experts",
        "Legal, Marketing and Technical experts analyse an idea in parallel; a Synthesizer merges the results.",
    )
    client = get_chat_client()

    # The experts: same input, different perspectives.
    def expert(name: str, focus: str) -> Agent:
        return Agent(
            client=client,
            name=name,
            instructions=f"You are a {focus} expert. Give your top 3 points about the idea in under 80 words.",
        )

    experts = [
        expert("LegalExpert", "legal and compliance"),
        expert("MarketingExpert", "go-to-market and marketing"),
        expert("TechExpert", "software architecture and engineering"),
    ]

    synthesizer = Agent(
        client=client,
        name="Synthesizer",
        instructions="Merge expert opinions into one balanced recommendation: Verdict, Top risks, Next steps.",
    )

    # ---- Custom aggregator (fan-in step) ---------------------------------
    async def synthesize(results: list[AgentExecutorResponse]) -> str:
        # Each item holds one expert's AgentResponse.
        sections = []
        for item in results:
            print_agent(item.executor_id, item.agent_response.text)
            sections.append(f"## {item.executor_id}\n{item.agent_response.text}")
        merged = await synthesizer.run("Expert opinions:\n\n" + "\n\n".join(sections))
        return merged.text

    workflow = ConcurrentBuilder(participants=experts).with_aggregator(synthesize).build()

    idea = "A mobile app that uses AI to analyse photos of skin moles and estimate cancer risk."
    print_agent("User", idea)

    print_section("Expert opinions (computed in parallel)")
    started = time.perf_counter()
    result = await workflow.run(idea)
    elapsed = time.perf_counter() - started

    print_section(f"Synthesized recommendation (total {elapsed:.1f}s)")
    for output in result.get_outputs():
        print_agent(synthesizer.name or "Synthesizer", str(output))


if __name__ == "__main__":
    asyncio.run(main())
