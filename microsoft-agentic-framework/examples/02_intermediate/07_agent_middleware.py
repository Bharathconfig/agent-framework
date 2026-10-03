"""
Example 07 - Agent & Chat Middleware  (Level: Intermediate)
==========================================================

GOAL
    Intercept every agent run and every model call to add cross-cutting
    behaviour (logging, timing, auditing) WITHOUT touching the agent itself.

WHAT YOU WILL LEARN
    The framework has three middleware layers, each wrapping the next:

        Agent middleware     -> wraps the whole agent.run() call
          Chat middleware    -> wraps EACH request to the LLM (there can be
                                several per run when tools are used)
            Function middleware -> wraps each tool call (see example 08)

    * Class-based middleware: subclass ``AgentMiddleware`` and implement ``process``.
    * Function-based middleware: decorate an async function with
      ``@chat_middleware`` (or ``@agent_middleware`` / ``@function_middleware``).
    * ``await call_next()`` continues the pipeline; code before it runs on the
      way IN, code after it runs on the way OUT (like an onion).
    * ``context.metadata`` lets middleware share data during a run.

RUN
    python examples/02_intermediate/07_agent_middleware.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import time  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402

from agent_framework import (  # noqa: E402
    Agent,
    AgentContext,
    AgentMiddleware,
    ChatContext,
    chat_middleware,
    tool,
)

from common import get_chat_client, print_agent, print_header  # noqa: E402


# ---------------------------------------------------------------------------
# 1) Class-based AGENT middleware: timing + audit log
# ---------------------------------------------------------------------------
class TimingAndAuditMiddleware(AgentMiddleware):
    """Measures how long each agent run takes and keeps a simple audit trail."""

    def __init__(self) -> None:
        # Middleware instances can hold state across runs (here: an audit list).
        self.audit_log: list[str] = []

    async def process(self, context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
        # ---- Runs BEFORE the agent ------------------------------------------
        last_user_text = context.messages[-1].text if context.messages else ""
        print(f"  [agent-mw] -> {context.agent.name} received: {last_user_text!r}")
        context.metadata["started"] = time.perf_counter()  # shared with later middleware

        # ---- Hand control to the next layer (eventually the agent itself) ---
        await call_next()

        # ---- Runs AFTER the agent -------------------------------------------
        elapsed = time.perf_counter() - context.metadata["started"]
        print(f"  [agent-mw] <- {context.agent.name} finished in {elapsed:.2f}s")
        self.audit_log.append(f"{context.agent.name}: {last_user_text[:40]!r} ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# 2) Function-based CHAT middleware: runs on EVERY call to the LLM
# ---------------------------------------------------------------------------
@chat_middleware
async def llm_call_logger(context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
    """Logs how many messages are sent to the model and what comes back."""
    print(f"  [chat-mw]  -> LLM request with {len(context.messages)} message(s)")
    await call_next()
    # context.result is a ChatResponse for non-streaming calls.
    result = context.result
    if result is not None and hasattr(result, "messages"):
        kinds = [c.type for m in result.messages for c in m.contents]
        print(f"  [chat-mw]  <- LLM returned content types: {kinds}")


@tool
def get_stock_price(symbol: str) -> str:
    """Return a (fake) current stock price for a ticker symbol."""
    return f"{symbol.upper()} is trading at $123.45"


async def main() -> None:
    print_header(
        "07 - Agent & Chat Middleware",
        "Watch the log lines: agent middleware wraps the run once, chat middleware wraps every LLM request.",
    )

    timing = TimingAndAuditMiddleware()

    agent = Agent(
        client=get_chat_client(),
        name="FinanceAgent",
        instructions="You answer stock questions. Use the get_stock_price tool.",
        tools=[get_stock_price],
        # Middleware order = outermost first.
        middleware=[timing, llm_call_logger],
    )

    for question in ["What's the price of MSFT?", "Explain what a stock is in one sentence."]:
        print_agent("User", question)
        response = await agent.run(question)
        print_agent(agent.name or "Agent", response.text)

    print("\nAudit log collected by the middleware:")
    for line in timing.audit_log:
        print(f"  - {line}")


if __name__ == "__main__":
    asyncio.run(main())
