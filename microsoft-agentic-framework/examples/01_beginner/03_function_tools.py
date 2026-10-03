"""
Example 03 - Function Tools  (Level: Beginner)
=============================================

GOAL
    Give the agent *tools* (plain Python functions) that it can decide to call
    to fetch real data or perform actions.

WHAT YOU WILL LEARN
    * Turning a Python function into a tool with the ``@tool`` decorator.
    * Describing parameters with ``Annotated[type, Field(description=...)]`` so
      the model knows how to call the tool correctly.
    * That the framework runs the *tool-calling loop* for you:
          model asks for tool -> framework executes Python -> result goes back
          to the model -> model writes the final answer.

HOW TOOL CALLING WORKS (behind the scenes)
    1. The framework sends the tool *schemas* (name, description, params) to the model.
    2. The model replies "please call get_weather(city='Paris')".
    3. The framework validates the arguments and calls your Python function.
    4. The function result is sent back to the model.
    5. Steps 2-4 repeat until the model produces a normal text answer.

RUN
    python examples/01_beginner/03_function_tools.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import random  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Annotated  # noqa: E402

from agent_framework import Agent, tool  # noqa: E402
from pydantic import Field  # noqa: E402

from common import get_chat_client, print_agent, print_header  # noqa: E402

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# The DOCSTRING becomes the tool description the model sees, so write it for
# the model: say what the tool does and when to use it.


@tool
def get_weather(
    city: Annotated[str, Field(description="Name of the city, e.g. 'Seattle'.")],
) -> str:
    """Get the current weather for a city."""
    # A real implementation would call a weather API. We fake it so the
    # example runs without any extra keys.
    conditions = ["sunny", "cloudy", "rainy", "windy"]
    return f"The weather in {city} is {random.choice(conditions)} and {random.randint(5, 30)} degrees Celsius."


@tool
def get_utc_time() -> str:
    """Get the current date and time in UTC (ISO-8601)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# The decorator also accepts options - here we override the name/description
# shown to the model instead of using the function name and docstring.
@tool(name="convert_currency", description="Convert an amount between two currencies using a fixed demo rate table.")
def convert(
    amount: Annotated[float, Field(description="Amount of money to convert.", gt=0)],
    source: Annotated[str, Field(description="ISO currency code to convert FROM, e.g. 'USD'.")],
    target: Annotated[str, Field(description="ISO currency code to convert TO, e.g. 'EUR'.")],
) -> str:
    # Demo rates relative to USD.
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.0, "JPY": 150.0}
    src, dst = source.upper(), target.upper()
    if src not in rates or dst not in rates:
        # Returning an error string (instead of raising) lets the model recover gracefully.
        return f"Unsupported currency. Supported: {', '.join(rates)}"
    value = amount / rates[src] * rates[dst]
    return f"{amount:.2f} {src} = {value:.2f} {dst}"


async def main() -> None:
    print_header(
        "03 - Function Tools",
        "The agent decides on its own which Python functions to call to answer the question.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="ToolAgent",
        instructions=(
            "You are a helpful travel assistant. Use the available tools to get "
            "real data - never guess weather, time or exchange rates."
        ),
        # Tools registered here are available on every run.
        tools=[get_weather, get_utc_time, convert],
    )

    question = "I'm flying to London. What's the weather there, what time is it in UTC, and how much is 250 USD in GBP?"
    print_agent("User", question)
    response = await agent.run(question)
    print_agent(agent.name or "Agent", response.text)

    # Bonus: inspect which tools were called during this run.
    calls = [c for m in response.messages for c in m.contents if c.type == "function_call"]
    print(f"\n(The model made {len(calls)} tool call(s): {', '.join(c.name for c in calls) or '-'})")


if __name__ == "__main__":
    asyncio.run(main())
