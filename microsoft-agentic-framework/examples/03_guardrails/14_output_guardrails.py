"""
Example 14 - Output Guardrails  (Level: Guardrails)
==================================================

GOAL
    Make sure the agent's ANSWER is safe before it reaches the user - even if
    the model or a tool returns something it shouldn't.

WHAT YOU WILL LEARN
    * ``OutputGuardrailMiddleware`` (in ``common/guardrails.py``) runs AFTER
      ``call_next()`` and can inspect/replace ``context.result``.
    * Typical output policies: redact PII leaks, withhold policy-violating
      content, enforce a maximum length.
    * Why output checks matter: tools often return raw data (customer
      records, logs, DB rows) that the model may repeat verbatim.

SCENARIO
    A CRM tool returns a full customer record including email, phone and
    card number. The agent is (deliberately) instructed to quote it. The
    output guardrail redacts the sensitive fields anyway.

RUN
    python examples/03_guardrails/14_output_guardrails.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent, tool  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402
from common.guardrails import OutputGuardrailMiddleware  # noqa: E402


@tool
def get_customer_record(customer_id: str) -> str:
    """Fetch a customer's record from the CRM."""
    # Simulated raw database row that contains sensitive data.
    return (
        f"id={customer_id}; name=Rahul Mehta; email=rahul.mehta@example.com; "
        "phone=+1 415 555 0134; card=5500 0000 0000 0004; tier=Gold; open_tickets=2"
    )


async def main() -> None:
    print_header(
        "14 - Output Guardrails",
        "The tool returns sensitive data; the output guardrail redacts it from the final answer.",
    )

    output_guard = OutputGuardrailMiddleware(max_chars=600)

    # The SAME agent configuration, with and without the guardrail, so you can compare.
    common_kwargs = {
        "client": get_chat_client(),
        "instructions": "You are a CRM assistant. When asked, quote the customer record exactly as returned.",
        "tools": [get_customer_record],
    }
    unguarded = Agent(name="UnguardedCRM", **common_kwargs)
    guarded = Agent(name="GuardedCRM", middleware=[output_guard], **common_kwargs)

    question = "Show me the full record for customer C-1042."

    print_section("Without output guardrail")
    print_agent("User", question)
    print_agent(unguarded.name or "Agent", (await unguarded.run(question)).text)

    print_section("With output guardrail")
    print_agent("User", question)
    print_agent(guarded.name or "Agent", (await guarded.run(question)).text)

    print_section("Guardrail audit trail")
    for event in output_guard.events or ["(no changes were necessary)"]:
        print(f"  - {event}")


if __name__ == "__main__":
    asyncio.run(main())
