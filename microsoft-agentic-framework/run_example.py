"""
run_example.py
==============

Convenience launcher so you never have to type long paths.

Usage
-----
    python run_example.py --list      # show all 25 examples
    python run_example.py 1           # run example 01
    python run_example.py 24          # run example 24
    python run_example.py all         # run every example one after another

You can of course still run any example directly, e.g.:

    python examples/01_beginner/01_hello_agent.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# The examples live in numbered sub-folders of ./examples
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def discover_examples() -> dict[int, Path]:
    """Return {example_number: path} for every "NN_*.py" file under ./examples."""
    found: dict[int, Path] = {}
    for path in sorted(EXAMPLES_DIR.glob("*/[0-9][0-9]_*.py")):
        found[int(path.name[:2])] = path
    return found


def list_examples(examples: dict[int, Path]) -> None:
    """Print a table of all examples grouped by level/folder."""
    current_group = None
    for number, path in examples.items():
        group = path.parent.name
        if group != current_group:
            print(f"\n{group}")
            current_group = group
        print(f"  {number:>2}  {path.stem}")
    print()


def run(path: Path, env: dict[str, str] | None = None) -> int:
    """Run one example in a fresh Python process and return its exit code."""
    print(f"\n>>> python {path.relative_to(EXAMPLES_DIR.parent)}\n", flush=True)
    return subprocess.call([sys.executable, str(path)], cwd=EXAMPLES_DIR.parent, env=env)


def main() -> int:
    examples = discover_examples()
    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help", "--list", "-l"}:
        print(__doc__)
        list_examples(examples)
        return 0

    arg = sys.argv[1].lower()
    if arg == "all":
        # Unattended mode: never block on input(). Example 10 auto-answers its
        # approval prompts ("no" = deny, unless you set AUTO_APPROVE yourself)
        # and example 22 uses its scripted customer replies.
        env = dict(os.environ)
        env.setdefault("AUTO_APPROVE", "no")
        env.pop("INTERACTIVE", None)
        # Stop at the first failing example so problems are easy to spot.
        for path in examples.values():
            code = run(path, env)
            if code != 0:
                return code
        return 0

    if not arg.isdigit() or int(arg) not in examples:
        print(f"Unknown example '{sys.argv[1]}'. Use --list to see the available numbers.")
        return 2
    return run(examples[int(arg)])


if __name__ == "__main__":
    sys.exit(main())
