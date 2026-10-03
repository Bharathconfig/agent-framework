"""
Example 22 - Multi-Agent: Handoff Customer Support  (Level: MAS)
===============================================================

GOAL
    A customer-support system in which a Triage agent hands the customer over
    to the right specialist (Billing or Technical), and specialists can hand
    back - with the human customer in the loop.

WHAT YOU WILL LEARN
    * ``HandoffBuilder`` - decentralised routing: agents decide themselves to
      transfer control by calling auto-generated ``handoff_to_<agent>`` tools.
    * ``.add_handoff(source, [targets])`` - define who may hand off to whom.
    * Human-in-the-loop in workflows: when an agent replies without handing
      off, the workflow emits a ``request_info`` event and PAUSES. We resume it
      with ``workflow.run(responses={request_id: answer})``.
    * Ending the conversation with ``HandoffAgentUserRequest.terminate()``.
    * Specialists still have their own tools.

TIP
    By default the customer replies are scripted. Set ``INTERACTIVE=1`` to
    type the customer's replies yourself.

RUN
    python examples/05_multi_agent/22_handoff_support.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import os  # noqa: E402

from agent_framework import Agent, tool  # noqa: E402
from agent_framework.orchestrations import HandoffAgentUserRequest, HandoffBuilder  # noqa: E402

from common import get_chat_client, print_agent, print_header  # noqa: E402


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by id."""
    return f"Invoice {invoice_id}: $49.00, charged twice on 2026-09-28 (duplicate charge detected)."


@tool
def issue_refund(invoice_id: str, amount: float) -> str:
    """Refund an amount for an invoice."""
    return f"Refund of ${amount:.2f} for {invoice_id} issued. Arrives in 3-5 business days."


@tool
def check_service_status(service: str) -> str:
    """Check whether a service is up."""
    return f"{service}: operational. No incidents in the last 24h."


# Scripted customer replies used when INTERACTIVE is not set.
SCRIPTED_REPLIES = [
    "The invoice id is INV-7781.",
    "Yes please refund the duplicate charge.",
    "Also, is the sync service working? My files aren't syncing.",
    "Thanks, that's all!",
]


def get_customer_reply(turn: int) -> str | None:
    """Return the next customer message, or None to end the conversation."""
    if os.getenv("INTERACTIVE"):
        text = input("\nCustomer (empty to finish): ").strip()
        return text or None
    if turn < len(SCRIPTED_REPLIES):
        print_agent("Customer", SCRIPTED_REPLIES[turn])
        return SCRIPTED_REPLIES[turn]
    return None


async def main() -> None:
    print_header(
        "22 - Handoff Support",
        "Triage routes the customer to Billing or Tech; the workflow pauses for customer replies.",
    )
    client = get_chat_client()

    triage = Agent(
        client=client,
        name="triage",
        # Required for handoff participants: keeps the local chat history in sync
        # when a handoff tool call short-circuits the agent's run.
        require_per_service_call_history_persistence=True,
        instructions=(
            "You are the front-desk support agent. Greet the customer briefly and hand off: billing/charges/"
            "refunds -> billing; outages/bugs/sync -> tech. Do not solve problems yourself."
        ),
    )
    billing = Agent(
        client=client,
        name="billing",
        # Required for handoff participants: keeps the local chat history in sync
        # when a handoff tool call short-circuits the agent's run.
        require_per_service_call_history_persistence=True,
        instructions=(
            "You handle billing. Use tools to look up invoices and issue refunds. Hand off tech issues to tech."
        ),
        tools=[lookup_invoice, issue_refund],
    )
    tech = Agent(
        client=client,
        name="tech",
        # Required for handoff participants: keeps the local chat history in sync
        # when a handoff tool call short-circuits the agent's run.
        require_per_service_call_history_persistence=True,
        instructions="You handle technical issues. Use check_service_status. Hand off billing issues to billing.",
        tools=[check_service_status],
    )

    workflow = (
        HandoffBuilder(participants=[triage, billing, tech])
        .with_start_agent(triage)  # first agent to receive the user's message
        .add_handoff(triage, [billing, tech])  # triage can route to both
        .add_handoff(billing, [tech])  # specialists can route to each other
        .add_handoff(tech, [billing])
        .build()
    )

    first_message = "Hi, I was charged twice this month!"
    print_agent("Customer", first_message)

    # Start the workflow. It runs until an agent needs the customer to reply.
    result = await workflow.run(first_message)
    turn = 0
    while True:
        # Show what the agents said and collect pending requests for the customer.
        pending = []
        for event in result:
            if event.type == "handoff_sent":
                print(f"  [handoff] {event.data.source} -> {event.data.target}")
            elif event.type == "request_info" and isinstance(event.data, HandoffAgentUserRequest):
                response = event.data.agent_response
                speaker = response.messages[-1].author_name if response.messages else "agent"
                print_agent(str(speaker or "agent"), response.text)
                pending.append(event.request_id)

        if not pending:
            break  # workflow finished

        reply = get_customer_reply(turn)
        turn += 1
        # Answer every pending request: a reply continues, terminate() ends the chat.
        answer = HandoffAgentUserRequest.create_response(reply) if reply else HandoffAgentUserRequest.terminate()
        result = await workflow.run(responses={request_id: answer for request_id in pending})

    print("\nConversation finished.")


if __name__ == "__main__":
    asyncio.run(main())
