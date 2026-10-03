"""
common/utils.py
===============

Small, dependency-free helpers that make the console output of the examples
easy to read. Nothing here is specific to the Agent Framework - it is purely
cosmetic so that you can follow what each example is doing.
"""

from __future__ import annotations

import textwrap

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
