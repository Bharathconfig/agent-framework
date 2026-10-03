"""
Example 02 - Streaming Responses  (Level: Beginner)
==================================================

GOAL
    Print the agent's answer token-by-token while it is being generated,
    exactly like ChatGPT does, instead of waiting for the full response.

WHAT YOU WILL LEARN
    * ``agent.run(..., stream=True)`` returns an *async iterator* of updates.
    * Each ``AgentResponseUpdate`` carries a small chunk of text (``.text``).
    * You can still get the complete, aggregated response at the end with
      ``await stream.get_final_response()``.
    * How to tune generation with ``default_options`` (temperature, max_tokens).

WHY IT MATTERS
    Streaming makes long answers feel instant and is essential for chat UIs.

RUN
    python examples/01_beginner/02_streaming_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent  # noqa: E402

from common import get_chat_client, print_header, print_section  # noqa: E402


async def main() -> None:
    print_header(
        "02 - Streaming Agent",
        "The agent writes a short story and we print each chunk as soon as it arrives.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="Storyteller",
        instructions="You are a creative storyteller who writes vivid but short stories.",
        # default_options are applied to EVERY run of this agent.
        #   temperature: 0 = deterministic, 1 = creative.
        #   max_tokens:  hard cap on the length of the answer (controls cost).
        # NOTE: reasoning models (o1/o3/o4/gpt-5) do not accept "temperature";
        # remove it if you deploy one of those.
        default_options={"temperature": 0.9, "max_tokens": 400},
    )

    print_section("Streaming output")

    # stream=True -> we get a ResponseStream instead of a finished response.
    stream = agent.run("Write a 5-sentence story about a robot learning to paint.", stream=True)

    # Iterate over the updates as they arrive from the model.
    async for update in stream:
        # Some updates carry no text (e.g. metadata), so guard against empty chunks.
        if update.text:
            # end="" keeps everything on one line; flush=True shows it immediately.
            print(update.text, end="", flush=True)
    print()

    # After the stream is consumed, the framework can give us the complete,
    # merged AgentResponse (same object you would get with stream=False).
    final = await stream.get_final_response()
    print_section("Final response statistics")
    print(f"Characters received : {len(final.text)}")
    # usage_details is filled when the service reports token usage.
    print(f"Token usage         : {final.usage_details or 'not reported'}")


if __name__ == "__main__":
    asyncio.run(main())
