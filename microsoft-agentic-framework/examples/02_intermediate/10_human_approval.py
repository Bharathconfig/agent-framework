"""
Example 10 - Human-in-the-Loop Tool Approval  (Level: Intermediate)
==================================================================

GOAL
    Let the agent PROPOSE sensitive actions (sending money, deleting data)
    but require a human to APPROVE them before they actually execute.

WHAT YOU WILL LEARN
    * ``@tool(approval_mode="always_require")`` marks a tool as sensitive.
    * When the model wants to call it, the run stops and the response contains
      ``response.user_input_requests`` (function-approval requests) instead of
      executing the tool.
    * You answer with ``request.to_function_approval_response(approved)`` and
      run the agent again with the SAME session; the framework then executes
      (or skips) the tool and the model finishes its answer.

TIP
    Set the environment variable ``AUTO_APPROVE=yes`` (or ``no``) to run this
    example non-interactively (e.g. in CI).

RUN
    python examples/02_intermediate/10_human_approval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import os  # noqa: E402
from typing import Annotated  # noqa: E402

from agent_framework import Agent, Message, tool  # noqa: E402
from pydantic import Field  # noqa: E402

from common import get_chat_client, print_agent, print_header  # noqa: E402

# A pretend bank account so we can see the effect of approved actions.
ACCOUNT = {"balance": 1_000.00}


@tool
def get_balance() -> str:
    """Return the current account balance. Safe - no approval needed."""
    return f"${ACCOUNT['balance']:.2f}"


@tool(approval_mode="always_require")  # <- the important line
def transfer_money(
    to: Annotated[str, Field(description="Recipient name")],
    amount: Annotated[float, Field(description="Amount in USD", gt=0)],
) -> str:
    """Transfer money to someone. REQUIRES HUMAN APPROVAL."""
    ACCOUNT["balance"] -= amount
    return f"Transferred ${amount:.2f} to {to}. New balance ${ACCOUNT['balance']:.2f}"


def ask_human(question: str) -> bool:
    """Ask the human operator (or use AUTO_APPROVE env var) for a yes/no answer."""
    preset = os.getenv("AUTO_APPROVE")
    if preset:
        print(f"{question} [auto: {preset}]")
        return preset.lower().startswith("y")
    try:
        return input(f"{question} [y/N]: ").strip().lower().startswith("y")
    except EOFError:  # no terminal attached -> safest default is to deny
        print("(no interactive input available - denying)")
        return False


async def main() -> None:
    print_header(
        "10 - Human Approval",
        "Sensitive tools pause the agent until a human approves or rejects the call.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="BankingAgent",
        instructions="You are a banking assistant. Use tools to check balances and make transfers.",
        tools=[get_balance, transfer_money],
    )
    # A session is required so the second run can see the pending tool call.
    session = agent.create_session()

    request_text = "Please send $150 to Alice for the concert tickets."
    print_agent("User", request_text)
    response = await agent.run(request_text, session=session)

    # Keep looping while the agent is waiting for approvals.
    while response.user_input_requests:
        approvals = []
        for request in response.user_input_requests:
            call = request.function_call  # the tool call the model wants to make
            approved = ask_human(f"\nAgent wants to call {call.name}({call.arguments}). Approve?")
            approvals.append(request.to_function_approval_response(approved))
        # Send the decisions back as a user message and continue the run.
        response = await agent.run(Message(role="user", contents=approvals), session=session)

    print_agent(agent.name or "Agent", response.text)
    print(f"\nFinal balance in our fake bank: ${ACCOUNT['balance']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
