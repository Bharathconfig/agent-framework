"""
Example 18 - Workflows: Conditional Routing  (Level: Workflows)
==============================================================

GOAL
    Route each incoming support email down a different path depending on an
    AI classification: spam is dropped, urgent issues get an escalation reply,
    everything else gets a standard reply.

WHAT YOU WILL LEARN
    * Wrapping an agent INSIDE a custom executor (full control over input,
      output type and structured output).
    * Passing typed messages (a dataclass) between nodes.
    * ``add_switch_case_edge_group`` with ``Case(condition, target)`` and
      ``Default(target)`` - like a ``match`` statement for graphs.
    * Running one workflow many times (one email per run).

        classify ──┬── Case(spam)   ──> SpamFilter      (python only)
                   ├── Case(urgent) ──> UrgentResponder (agent)
                   └── Default      ──> StandardResponder (agent)

RUN
    python examples/04_workflows/18_conditional_routing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Literal, Never  # noqa: E402

from agent_framework import Agent, Case, Default, Executor, WorkflowBuilder, WorkflowContext, handler  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402


# ---------------------------------------------------------------------------
# Data passed between nodes
# ---------------------------------------------------------------------------
class TriageResult(BaseModel):
    """Structured output of the classifier agent."""

    category: Literal["spam", "urgent", "normal"] = Field(
        description="spam = unsolicited/phishing; urgent = outage, security, payment failure; otherwise normal"
    )
    reason: str = Field(description="One short sentence explaining the category")


@dataclass
class TriagedEmail:
    """Message sent from the classifier to the next node."""

    email: str
    category: str
    reason: str


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
class Classifier(Executor):
    """Uses an agent with structured output to categorise the email."""

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="classifier")
        self.agent = agent

    @handler
    async def classify(self, email: str, ctx: WorkflowContext[TriagedEmail]) -> None:
        result = await self.agent.run(email, options={"response_format": TriageResult})
        triage = result.value or TriageResult(category="normal", reason="classifier fallback")
        print(f"  [classifier] -> {triage.category}: {triage.reason}")
        await ctx.send_message(TriagedEmail(email=email, category=triage.category, reason=triage.reason))


class SpamFilter(Executor):
    """Pure Python branch - no LLM cost for spam."""

    def __init__(self) -> None:
        super().__init__(id="spam_filter")

    @handler
    async def drop(self, item: TriagedEmail, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"[SPAM - discarded] reason: {item.reason}")


class Responder(Executor):
    """Generic agent-backed responder used for the urgent and standard branches."""

    def __init__(self, id: str, agent: Agent, prefix: str) -> None:
        super().__init__(id=id)
        self.agent = agent
        self.prefix = prefix

    @handler
    async def reply(self, item: TriagedEmail, ctx: WorkflowContext[Never, str]) -> None:
        response = await self.agent.run(f"Customer email:\n{item.email}")
        await ctx.yield_output(f"{self.prefix}\n{response.text}")


async def main() -> None:
    print_header(
        "18 - Conditional Routing",
        "A classifier agent decides which branch of the workflow handles each email.",
    )
    client = get_chat_client()

    classifier = Classifier(
        Agent(client=client, name="Triage", instructions="Classify customer emails.", default_options={"temperature": 0})
    )
    spam = SpamFilter()
    urgent = Responder(
        "urgent_responder",
        Agent(client=client, name="Escalation", instructions="Write a calm, 3-sentence reply promising a call within 1 hour."),
        prefix="[URGENT - escalated to on-call]",
    )
    standard = Responder(
        "standard_responder",
        Agent(client=client, name="Support", instructions="Write a friendly 2-sentence support reply."),
        prefix="[STANDARD reply]",
    )

    workflow = (
        WorkflowBuilder(start_executor=classifier, output_from=[spam, urgent, standard])
        .add_switch_case_edge_group(
            classifier,
            [
                # Conditions receive the message sent by the source node (TriagedEmail).
                Case(condition=lambda m: m.category == "spam", target=spam),
                Case(condition=lambda m: m.category == "urgent", target=urgent),
                Default(target=standard),  # anything else
            ],
        )
        .build()
    )

    emails = [
        "CONGRATULATIONS!!! You won $1,000,000. Click this link and enter your bank password to claim.",
        "Our production checkout is DOWN and every payment fails since 10 minutes. Please help immediately!",
        "Hi, could you tell me how to change the email address on my account? Thanks.",
    ]
    for email in emails:
        print_section(email[:60] + "...")
        result = await workflow.run(email)  # non-streaming: returns all events
        for output in result.get_outputs():
            print_agent("Workflow", str(output))


if __name__ == "__main__":
    asyncio.run(main())
