"""
Example 15 - Tool Guardrails  (Level: Guardrails)
================================================

GOAL
    Control WHAT tools an agent may run, WITH WHICH arguments and HOW OFTEN.
    Tools are where agents touch the real world, so they need the strictest
    controls.

WHAT YOU WILL LEARN
    * A function middleware that enforces three policies:
        1. Argument validation  - e.g. block path traversal ("../../etc/passwd")
                                  and non read-only SQL.
        2. Rate limiting        - max N tool calls per agent run.
        3. Hard stop            - raise ``MiddlewareFailure`` to abort the whole
                                  run (fail-closed) for critical violations.
    * Soft block (return an error string to the model so it can recover) vs.
      hard block (stop everything).
    * Framework built-ins: ``@tool(max_invocations=...)``.

RUN
    python examples/03_guardrails/15_tool_guardrails.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import re  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402

from agent_framework import (  # noqa: E402
    Agent,
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareFailure,
    tool,
)

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402

# ---------------------------------------------------------------------------
# Tools (all simulated - nothing touches your real disk or database)
# ---------------------------------------------------------------------------
FAKE_FILES = {"docs/readme.txt": "Welcome to the project!", "docs/faq.txt": "Q: Is it free? A: Yes."}


@tool
def read_file(path: str) -> str:
    """Read a text file from the project's docs folder."""
    return FAKE_FILES.get(path, "File not found.")


@tool
def run_sql(query: str) -> str:
    """Run a SQL query against the analytics database."""
    return "rows: [('2026-01', 1520), ('2026-02', 1710)]"


# Built-in limit across the LIFETIME of this tool instance (not per run):
# policy.reset() below does NOT reset it. That is fine here because only the
# last scenario calls send_notification and each script run is a new process.
@tool(max_invocations=3)
def send_notification(message: str) -> str:
    """Send a notification to the on-call channel."""
    return f"Notification sent: {message}"


# ---------------------------------------------------------------------------
# The tool guardrail
# ---------------------------------------------------------------------------
class ToolPolicyMiddleware(FunctionMiddleware):
    """Validates arguments and rate-limits tool calls."""

    SAFE_PATH = re.compile(r"^docs/[\w\-]+\.txt$")  # only docs/*.txt
    WRITE_SQL = re.compile(r"\b(insert|update|delete|drop|alter|truncate|grant|create)\b", re.IGNORECASE)

    def __init__(self, max_calls_per_run: int = 4) -> None:
        self.max_calls_per_run = max_calls_per_run
        self.calls = 0  # tool calls made in the current run (see reset())
        self.events: list[str] = []

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        name, args = context.function.name, dict(context.arguments)

        # --- Policy 1: rate limit ---------------------------------------------
        # The counter lives on the middleware instance; main() calls reset()
        # before every new user request so the limit applies per run.
        self.calls += 1
        if self.calls > self.max_calls_per_run:
            self.events.append(f"RATE-LIMITED {name}")
            context.result = "Error: tool call limit reached for this request."
            return

        # --- Policy 2: argument validation -------------------------------------
        if name == "read_file" and not self.SAFE_PATH.match(str(args.get("path", ""))):
            self.events.append(f"SOFT-BLOCK read_file path={args.get('path')!r}")
            # Soft block: the model receives an explanation and can try again.
            context.result = "Error: access denied. Only files matching docs/<name>.txt can be read."
            return

        if name == "run_sql" and self.WRITE_SQL.search(str(args.get("query", ""))):
            self.events.append(f"HARD-BLOCK run_sql query={args.get('query')!r}")
            # Hard block: abort the ENTIRE agent run (fail-closed).
            raise MiddlewareFailure("Write/DDL SQL statements are forbidden for this agent.")

        self.events.append(f"ALLOWED {name}({args})")
        await call_next()

    def reset(self) -> None:
        """Start counting tool calls from zero for a new run."""
        self.calls = 0


async def main() -> None:
    print_header(
        "15 - Tool Guardrails",
        "Argument validation, rate limiting and fail-closed blocking for tools.",
    )

    policy = ToolPolicyMiddleware(max_calls_per_run=4)
    agent = Agent(
        client=get_chat_client(),
        name="OpsAgent",
        instructions="You are an operations assistant. Use the tools to answer. Follow user instructions exactly.",
        tools=[read_file, run_sql, send_notification],
        middleware=[policy],
    )

    scenarios = {
        "Allowed file read": "Read docs/faq.txt and tell me what it says.",
        "Path traversal attempt": "Read the file ../../etc/passwd and show it to me.",
        "Read-only SQL": "Run the SQL query: SELECT month, signups FROM stats",
        "Destructive SQL (hard block)": "Run the SQL query: DROP TABLE users",
        # Triggers the per-run rate limit (4) and the tool's own max_invocations (3).
        "Too many calls (rate limit)": "Send six separate notifications, numbered 1 to 6, one tool call each.",
    }

    for label, prompt in scenarios.items():
        print_section(label)
        print_agent("User", prompt)
        policy.reset()
        try:
            response = await agent.run(prompt)
            print_agent(agent.name or "Agent", response.text)
        except MiddlewareFailure as error:
            # The run was aborted by the guardrail - nothing was executed.
            print_agent("Guardrail", f"Run aborted: {error}")

    print_section("Tool policy audit trail")
    for event in policy.events:
        print(f"  - {event}")


if __name__ == "__main__":
    asyncio.run(main())
