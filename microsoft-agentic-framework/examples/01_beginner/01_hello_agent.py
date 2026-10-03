"""
Example 01 - Hello Agent  (Level: Beginner)
==========================================

GOAL
    Create the simplest possible AI agent with the Microsoft Agent Framework
    and ask it one question.

WHAT YOU WILL LEARN
    * What a *chat client* is (the connection to the LLM).
    * What an *Agent* is (client + instructions + optional tools/memory).
    * How to call ``agent.run()`` and read ``response.text``.

KEY CONCEPTS
    Chat client  -> knows HOW to talk to a model (Azure OpenAI here).
    Agent        -> knows WHAT to do: it has a name, instructions (system
                    prompt) and can later get tools, memory, middleware, ...
    agent.run()  -> sends your message, waits for the model, returns an
                    ``AgentResponse``. ``.text`` is the final assistant text.

RUN
    python examples/01_beginner/01_hello_agent.py
"""

# ---------------------------------------------------------------------------
# Bootstrap: make the shared "common" package (in the project root) importable
# when this file is executed directly as a script.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The Agent Framework is fully asynchronous, so we use asyncio to run it.
import asyncio  # noqa: E402

# "Agent" is the main building block of the framework.
from agent_framework import Agent  # noqa: E402

# get_chat_client() reads your .env and returns an Azure OpenAI chat client.
from common import get_chat_client, print_agent, print_header  # noqa: E402


async def main() -> None:
    print_header(
        "01 - Hello Agent",
        "A single agent with a name and instructions answers one question.",
    )

    # 1) Create the chat client. This holds the endpoint, key and deployment.
    client = get_chat_client()

    # 2) Create the agent.
    #    - name:         used in logs / multi-agent conversations.
    #    - instructions: the "system prompt" - the agent's job description.
    agent = Agent(
        client=client,
        name="HelloAgent",
        instructions="You are a friendly assistant. Answer in at most three sentences.",
    )

    # 3) Ask a question. "await" because the call goes over the network.
    question = "What is an AI agent, in simple words?"
    print_agent("User", question)
    response = await agent.run(question)

    # 4) The AgentResponse contains all messages produced in this run.
    #    .text concatenates the assistant text, which is usually all you need.
    print_agent(agent.name or "Agent", response.text)


if __name__ == "__main__":
    # asyncio.run() starts the event loop, runs main() and closes the loop.
    asyncio.run(main())
