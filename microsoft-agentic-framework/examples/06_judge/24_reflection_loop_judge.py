"""
Example 24 - Judge Agent: Writer <-> Judge Reflection Loop  (Level: Judge)
=========================================================================

GOAL
    Iteratively improve an answer: a Writer drafts, a Judge scores it and gives
    concrete feedback, the Writer revises - until the Judge approves or the
    maximum number of rounds is reached.

WHAT YOU WILL LEARN
    * Building a CYCLIC workflow graph (writer -> judge -> writer -> ...).
    * Using typed messages (dataclasses) to carry drafts and feedback.
    * Stopping conditions for loops: quality threshold AND an iteration cap
      (never build an agent loop without a hard limit!).
    * Keeping the writer's memory between rounds with an ``AgentSession`` so
      it revises its own draft instead of starting from scratch.

        ┌──────────┐  Draft   ┌─────────┐
        │  Writer  │ ───────> │  Judge  │ ──(approved / max rounds)──> output
        └──────────┘ <─────── └─────────┘
                     Feedback

RUN
    python examples/06_judge/24_reflection_loop_judge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402

PASS_SCORE = 8  # judge score (1-10) needed for approval
MAX_ROUNDS = 3  # hard cap on revisions


# ---------------------------------------------------------------------------
# Messages exchanged between the two nodes
# ---------------------------------------------------------------------------
@dataclass
class Draft:
    text: str
    round: int


@dataclass
class Feedback:
    score: int
    critique: str
    round: int


class Evaluation(BaseModel):
    """Structured judgement returned by the judge agent."""

    score: int = Field(description="Overall quality 1-10")
    strengths: list[str] = Field(description="What is good")
    improvements: list[str] = Field(description="Specific, actionable changes required to reach a 9-10")


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
class WriterExecutor(Executor):
    """Writes the first draft from the task, then revises based on feedback."""

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="writer")
        self.agent = agent
        self.session = agent.create_session()  # remembers previous drafts

    @handler
    async def write_first_draft(self, task: str, ctx: WorkflowContext[Draft]) -> None:
        response = await self.agent.run(task, session=self.session)
        print_agent("Writer (round 1)", response.text)
        await ctx.send_message(Draft(text=response.text, round=1))

    @handler
    async def revise(self, feedback: Feedback, ctx: WorkflowContext[Draft]) -> None:
        prompt = (
            f"The judge scored your draft {feedback.score}/10. Revise it addressing ALL of this feedback:\n"
            f"{feedback.critique}\nReturn only the improved text."
        )
        response = await self.agent.run(prompt, session=self.session)
        next_round = feedback.round + 1
        print_agent(f"Writer (round {next_round})", response.text)
        await ctx.send_message(Draft(text=response.text, round=next_round))


class JudgeExecutor(Executor):
    """Scores each draft; either approves (yields output) or sends feedback."""

    def __init__(self, agent: Agent) -> None:
        super().__init__(id="judge")
        self.agent = agent

    @handler
    async def evaluate(self, draft: Draft, ctx: WorkflowContext[Feedback, str]) -> None:
        response = await self.agent.run(
            f"Evaluate this draft:\n\n{draft.text}", options={"response_format": Evaluation}
        )
        evaluation: Evaluation | None = response.value
        score = evaluation.score if evaluation else 0
        improvements = evaluation.improvements if evaluation else ["Judge output was invalid; tighten the text."]
        print(f"\n  [judge] round {draft.round}: score {score}/10")
        for item in improvements:
            print(f"    - {item}")

        if score >= PASS_SCORE:
            await ctx.yield_output(f"APPROVED in round {draft.round} with score {score}/10:\n\n{draft.text}")
        elif draft.round >= MAX_ROUNDS:
            await ctx.yield_output(
                f"STOPPED after {MAX_ROUNDS} rounds (best effort, last score {score}/10):\n\n{draft.text}"
            )
        else:
            # Loop back to the writer.
            await ctx.send_message(Feedback(score=score, critique="\n".join(improvements), round=draft.round))


async def main() -> None:
    print_header(
        "24 - Reflection Loop with a Judge",
        f"Writer and Judge iterate until score >= {PASS_SCORE}/10 or {MAX_ROUNDS} rounds.",
    )
    client = get_chat_client()

    writer = WriterExecutor(
        Agent(client=client, name="Writer", instructions="You are a technical writer. Write clear, accurate text.")
    )
    judge = JudgeExecutor(
        Agent(
            client=client,
            name="Judge",
            instructions=(
                "You are a demanding editor. Evaluate accuracy, clarity, structure and audience fit. "
                "Only give 9-10 to text that needs no changes."
            ),
            default_options={"temperature": 0},
        )
    )

    workflow = (
        WorkflowBuilder(start_executor=writer, output_from=[judge], max_iterations=20)
        .add_edge(writer, judge)  # Draft    -> judge
        .add_edge(judge, writer)  # Feedback -> writer (the loop)
        .build()
    )

    task = "Explain to a non-technical manager, in about 120 words, what an API is and why it matters for our business."
    print_agent("User", task)
    print_section("Iterations")
    result = await workflow.run(task)

    print_section("Result")
    for output in result.get_outputs():
        print_agent("Workflow", str(output))


if __name__ == "__main__":
    asyncio.run(main())
