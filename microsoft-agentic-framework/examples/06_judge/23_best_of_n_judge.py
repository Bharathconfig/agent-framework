"""
Example 23 - Judge Agent: Best-of-N Selection  (Level: Judge / Evaluation)
=========================================================================

GOAL
    Generate several candidate answers IN PARALLEL with different agents,
    then let an impartial JUDGE agent score every candidate against a rubric
    and pick the winner.

WHAT YOU WILL LEARN
    * The "LLM-as-a-judge" pattern: a separate agent evaluates other agents'
      output using explicit criteria and returns STRUCTURED scores.
    * Combining patterns: ``ConcurrentBuilder`` (fan-out generation) + a
      custom aggregator that calls the judge (fan-in evaluation).
    * Designing a good judge: clear rubric, anonymised candidates (to reduce
      bias), low temperature, score + justification for every criterion.

WHY IT MATTERS
    Best-of-N with a judge reliably improves quality for creative or
    open-ended tasks, and the scores give you an automatic quality signal you
    can log, monitor or alert on.

RUN
    python examples/06_judge/23_best_of_n_judge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402

from agent_framework import Agent, AgentExecutorResponse  # noqa: E402
from agent_framework.orchestrations import ConcurrentBuilder  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import get_chat_client, print_agent, print_header, print_section  # noqa: E402


# ---------------------------------------------------------------------------
# The judge's structured output (the "rubric")
# ---------------------------------------------------------------------------
class CandidateScore(BaseModel):
    candidate: str = Field(description="Candidate label exactly as given, e.g. 'Candidate A'")
    relevance: int = Field(description="How well it answers the brief (1-10)")
    clarity: int = Field(description="How clear and easy to understand (1-10)")
    persuasiveness: int = Field(description="How compelling it is for the audience (1-10)")
    justification: str = Field(description="1-2 sentences explaining the scores")


class JudgeReport(BaseModel):
    scores: list[CandidateScore]
    winner: str = Field(description="Label of the best candidate")
    rationale: str = Field(description="Why the winner is better than the others")


JUDGE_INSTRUCTIONS = """
You are an impartial expert judge. You will receive a BRIEF and several anonymous CANDIDATES.
Score EVERY candidate on relevance, clarity and persuasiveness (integers 1-10).
Rules:
- Judge only the content, not the length or position of a candidate.
- Be strict: 9-10 only for truly excellent work.
- The winner must be the candidate with the best overall quality.
"""


async def main() -> None:
    print_header(
        "23 - Best-of-N with a Judge",
        "Three copywriters write in parallel; a judge agent scores them with a rubric and picks the winner.",
    )
    client = get_chat_client()

    brief = "Write a 2-sentence tagline + pitch for a reusable smart water bottle that tracks hydration."

    # --- N candidate generators: different styles for diversity --------------
    writers = [
        Agent(client=client, name="Minimalist", instructions="You write short, elegant, minimalist copy."),
        Agent(client=client, name="Storyteller", instructions="You write emotional, story-driven copy."),
        Agent(client=client, name="DataDriven", instructions="You write copy that leads with numbers and benefits."),
    ]

    judge = Agent(
        client=client,
        name="Judge",
        instructions=JUDGE_INSTRUCTIONS,
        default_options={"temperature": 0},  # consistent, repeatable judgments
    )

    # --- Aggregator = judging step -------------------------------------------
    async def judge_candidates(results: list[AgentExecutorResponse]) -> str:
        # Anonymise: the judge sees "Candidate A/B/C", not the writer's name.
        labels = {f"Candidate {chr(65 + i)}": r for i, r in enumerate(results)}
        for label, r in labels.items():
            print_agent(f"{label} (by {r.executor_id})", r.agent_response.text)

        prompt = f"BRIEF:\n{brief}\n\n" + "\n\n".join(
            f"{label}:\n{r.agent_response.text}" for label, r in labels.items()
        )
        report: JudgeReport | None = (await judge.run(prompt, options={"response_format": JudgeReport})).value
        if report is None:
            return "The judge did not return a valid report."

        print_section("Judge scoreboard")
        print(f"  {'Candidate':<12} {'Rel':>4} {'Clar':>5} {'Pers':>5} {'Total':>6}  Justification")
        for s in report.scores:
            total = s.relevance + s.clarity + s.persuasiveness
            scores = f"{s.relevance:>4} {s.clarity:>5} {s.persuasiveness:>5} {total:>6}"
            print(f"  {s.candidate:<12} {scores}  {s.justification}")

        winner = labels.get(report.winner)
        author = winner.executor_id if winner else "unknown"
        winning_text = winner.agent_response.text if winner else "(winner label not recognised)"
        return f"WINNER: {report.winner} (written by {author})\n{winning_text}\n\nWhy: {report.rationale}"

    workflow = ConcurrentBuilder(participants=writers).with_aggregator(judge_candidates).build()

    print_agent("User (brief)", brief)
    print_section("Candidates")
    result = await workflow.run(brief)

    print_section("Final decision")
    for output in result.get_outputs():
        print_agent("Judge", str(output))


if __name__ == "__main__":
    asyncio.run(main())
