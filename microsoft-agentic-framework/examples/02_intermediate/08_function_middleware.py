"""
Example 08 - Function (Tool) Middleware  (Level: Intermediate)
=============================================================

GOAL
    Wrap tool calls with reusable logic: logging, caching, retries and
    result post-processing.

WHAT YOU WILL LEARN
    * ``FunctionMiddleware`` sees every tool call: ``context.function`` (the
      tool), ``context.arguments`` (validated args) and ``context.result``.
    * Short-circuiting: set ``context.result`` and DON'T call ``call_next()``
      to skip the real tool (used here for a cache).
    * Retrying a flaky tool by calling ``call_next()`` more than once.
    * Post-processing a result after ``call_next()``.

RUN
    python examples/02_intermediate/08_function_middleware.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402

from agent_framework import Agent, FunctionInvocationContext, FunctionMiddleware, function_middleware, tool  # noqa: E402

from common import get_chat_client, print_agent, print_header  # noqa: E402


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def lookup_population(country: str) -> str:
    """Look up the (approximate) population of a country."""
    data = {"india": "1.44 billion", "japan": "124 million", "brazil": "216 million", "france": "68 million"}
    return data.get(country.lower(), "unknown")


@tool
def flaky_exchange_rate(currency: str) -> str:
    """Get the USD exchange rate for a currency. (Simulates an unreliable API.)"""
    # Fails ~50% of the time to demonstrate the retry middleware.
    if random.random() < 0.5:
        raise ConnectionError("Upstream API timed out")
    return f"1 USD = {random.uniform(0.5, 100):.2f} {currency.upper()}"


def result_as_text(result: object) -> str:
    """Tool results are usually a list of Content items - turn them into readable text."""
    if isinstance(result, list):
        return " ".join(str(getattr(item, "text", None) or item) for item in result)
    return str(result)


# ---------------------------------------------------------------------------
# Middleware 1: logging (function-based)
# ---------------------------------------------------------------------------
@function_middleware
async def log_tool_calls(context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
    print(f"  [tool-log]   calling {context.function.name}({dict(context.arguments)})")
    await call_next()
    print(f"  [tool-log]   {context.function.name} returned -> {result_as_text(context.result)}")


# ---------------------------------------------------------------------------
# Middleware 2: cache (class-based, stateful)
# ---------------------------------------------------------------------------
class CacheMiddleware(FunctionMiddleware):
    """Caches tool results by (tool name, arguments)."""

    def __init__(self, cacheable: set[str]) -> None:
        self.cacheable = cacheable  # only cache tools whose results don't change
        self.cache: dict[str, object] = {}

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if context.function.name not in self.cacheable:
            await call_next()
            return
        key = f"{context.function.name}:{json.dumps(dict(context.arguments), sort_keys=True)}"
        if key in self.cache:
            print(f"  [cache]      HIT  {key}")
            context.result = self.cache[key]  # short-circuit: real tool is NOT executed
            return
        print(f"  [cache]      MISS {key}")
        await call_next()
        self.cache[key] = context.result


# ---------------------------------------------------------------------------
# Middleware 3: retry with back-off
# ---------------------------------------------------------------------------
class RetryMiddleware(FunctionMiddleware):
    """Retries a failing tool up to ``max_attempts`` times."""

    def __init__(self, max_attempts: int = 4, delay_seconds: float = 0.2) -> None:
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        for attempt in range(1, self.max_attempts + 1):
            try:
                await call_next()
                return
            except Exception as error:  # noqa: BLE001 - we want to retry any tool failure
                print(f"  [retry]      attempt {attempt} failed: {error}")
                if attempt == self.max_attempts:
                    # Give the model a clean, safe message instead of a stack trace.
                    context.result = f"Tool unavailable after {attempt} attempts. Tell the user to try later."
                    return
                await asyncio.sleep(self.delay_seconds * attempt)  # linear back-off


async def main() -> None:
    print_header(
        "08 - Function Middleware",
        "Logging, caching and retrying tool calls with composable middleware.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="DataAgent",
        instructions="Answer using the tools. Always call a tool for populations and exchange rates.",
        tools=[lookup_population, flaky_exchange_rate],
        # Order matters: log (outermost) -> cache -> retry (innermost, closest to the tool).
        middleware=[log_tool_calls, CacheMiddleware(cacheable={"lookup_population"}), RetryMiddleware()],
    )

    # Same session so the second question re-uses the cache.
    session = agent.create_session()
    for question in [
        "What is the population of Japan?",
        "Tell me Japan's population again, and the USD to JPY exchange rate.",
    ]:
        print_agent("User", question)
        response = await agent.run(question, session=session)
        print_agent(agent.name or "Agent", response.text)


if __name__ == "__main__":
    asyncio.run(main())
