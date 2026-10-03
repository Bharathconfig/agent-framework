"""
common/utils.py
===============

Small, dependency-free helpers that make the console output of the examples
easy to read. Nothing here is specific to the Agent Framework - it is purely
cosmetic so that you can follow what each example is doing.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable

# Width used for the decorative separators.
_WIDTH = 78


def print_header(title: str, description: str = "") -> None:
    """Print a big banner at the start of an example.

    Args:
        title: Short title of the example (e.g. "01 - Hello Agent").
        description: Optional one-paragraph explanation printed under the title.
    """
    print("=" * _WIDTH)
    print(f" {title}")
    print("=" * _WIDTH)
    if description:
        print(textwrap.fill(description.strip(), width=_WIDTH))
        print("-" * _WIDTH)


def print_section(title: str) -> None:
    """Print a smaller sub-heading inside an example."""
    print(f"\n--- {title} " + "-" * max(0, _WIDTH - len(title) - 5))


def print_agent(name: str, text: str) -> None:
    """Pretty-print a message produced by an agent.

    Args:
        name: Who is speaking (agent name, "User", "Judge", ...).
        text: What they said.
    """
    print(f"\n[{name}]")
    print(textwrap.indent(text.strip() or "(empty response)", "  "))


def print_workflow_output(output: object) -> None:
    """Print whatever a multi-agent workflow yielded, in a readable way.

    Orchestrations yield different shapes depending on the pattern:

    * ``list[Message]``  - the full conversation (sequential, group chat, handoff)
    * ``AgentResponse``  - an object with ``.messages`` (concurrent default aggregator)
    * ``str`` or anything else - printed as-is
    """
    messages = getattr(output, "messages", output)
    if isinstance(messages, list) and all(hasattr(m, "role") for m in messages):
        for message in messages:
            if not message.text.strip():
                continue  # skip tool-call-only messages
            speaker = message.author_name or message.role
            print_agent(str(speaker), message.text)
    else:
        print_agent("Output", str(output))


def print_workflow_events(events: Iterable[object]) -> None:
    """Print every agent turn contained in a (non-streaming) workflow result.

    ``await workflow.run(...)`` returns a ``WorkflowRunResult`` - a list of
    ``WorkflowEvent`` objects. Agent turns surface as ``intermediate`` events
    (when the orchestration was built with ``intermediate_output_from=...``)
    and the final answer as an ``output`` event. ``event.executor_id`` tells
    us which agent/node produced it.
    """
    for event in events:
        if getattr(event, "type", None) not in {"intermediate", "output"}:
            continue
        data = event.data  # type: ignore[attr-defined]
        label = event.executor_id or "workflow"  # type: ignore[attr-defined]
        if getattr(event, "type", None) == "output":
            label = f"{label} (final output)"
        text = getattr(data, "text", None)
        if text is None:
            print_workflow_output(data)
        else:
            print_agent(label, text)
