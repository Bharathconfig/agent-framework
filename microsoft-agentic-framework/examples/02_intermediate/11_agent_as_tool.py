"""
Example 11 - Agents as Tools (Supervisor Pattern)  (Level: Intermediate)
=======================================================================

GOAL
    Build a "manager" agent that delegates sub-tasks to specialist agents by
    calling them like ordinary tools. This is the simplest multi-agent system.

WHAT YOU WILL LEARN
    * ``agent.as_tool()`` wraps any agent as a ``FunctionTool``.
    * The supervisor decides WHICH specialist to call, WITH WHAT task, and
      combines their answers.
    * Specialists can have their own tools (the math expert has a calculator).

WHEN TO USE
    When one coordinator should stay in control and specialists only answer
    narrowly-scoped questions. For agents that talk to each other directly,
    see the multi-agent examples (19-25).

RUN
    python examples/02_intermediate/11_agent_as_tool.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ast  # noqa: E402
import asyncio  # noqa: E402
import operator  # noqa: E402

from agent_framework import Agent, tool  # noqa: E402

from common import get_chat_client, print_agent, print_header  # noqa: E402

# ---------------------------------------------------------------------------
# A SAFE calculator tool. Never use eval() on model output! We parse the
# expression into an AST and only allow basic arithmetic operators.
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_evaluate(node.right)) > 100:
            raise ValueError("exponent too large")  # avoid huge computations
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression like '(12.5 * 4) / 3'."""
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        return f"Error: {error}"


async def main() -> None:
    print_header(
        "11 - Agents as Tools",
        "A supervisor agent delegates to a math expert and a writing expert.",
    )
    client = get_chat_client()

    # --- Specialist 1: math expert with a calculator tool -------------------
    math_expert = Agent(
        client=client,
        name="MathExpert",
        description="Solves numeric and math problems precisely.",  # shown to the supervisor
        instructions="You solve math problems. ALWAYS use the calculate tool for arithmetic. Show the result.",
        tools=[calculate],
    )

    # --- Specialist 2: writing expert ---------------------------------------
    writer = Agent(
        client=client,
        name="WritingExpert",
        description="Writes and polishes short texts such as emails and announcements.",
        instructions="You write clear, friendly, professional short texts.",
    )

    # --- Supervisor -----------------------------------------------------------
    supervisor = Agent(
        client=client,
        name="Supervisor",
        instructions=(
            "You coordinate specialists. Break the user's request into parts, call the "
            "math_expert tool for calculations and the writing_expert tool for writing. "
            "Then return the final combined answer."
        ),
        tools=[
            # as_tool() turns the agent into a tool. 'task' is the argument the
            # supervisor fills in with the sub-task description.
            math_expert.as_tool(name="math_expert", arg_description="The math problem to solve"),
            writer.as_tool(name="writing_expert", arg_description="What to write, including all facts to use"),
        ],
    )

    request = (
        "Our team of 7 people sold 1,240 tickets at $18.50 each. Compute total revenue and revenue per person, "
        "then write a short celebratory email to the team with those numbers."
    )
    print_agent("User", request)
    response = await supervisor.run(request)
    print_agent(supervisor.name or "Agent", response.text)


if __name__ == "__main__":
    asyncio.run(main())
