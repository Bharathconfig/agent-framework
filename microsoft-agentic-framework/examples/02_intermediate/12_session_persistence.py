"""
Example 12 - Persisting and Resuming Sessions  (Level: Intermediate)
===================================================================

GOAL
    Save a conversation to disk and resume it later (after a restart, on
    another server, ...) so the agent continues exactly where it left off.

WHAT YOU WILL LEARN
    * ``session.to_dict()`` -> a JSON-serialisable snapshot of the session
      (including the chat history stored by ``InMemoryHistoryProvider``).
    * ``AgentSession.from_dict(data)`` -> rebuild the session.
    * A simple file-based store you could swap for Redis/Cosmos DB/SQL.

RUN
    python examples/02_intermediate/12_session_persistence.py
    (run it twice! the second time it resumes the saved conversation)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
import json  # noqa: E402

from agent_framework import Agent, AgentSession  # noqa: E402

from common import get_chat_client, print_agent, print_header  # noqa: E402
from common.config import PROJECT_ROOT  # noqa: E402

# Sessions are stored in "<project>/.sessions/" (git-ignored).
SESSION_DIR = PROJECT_ROOT / ".sessions"
SESSION_FILE = SESSION_DIR / "trip_planner.json"


def save_session(session: AgentSession) -> None:
    """Serialise the session to a JSON file."""
    SESSION_DIR.mkdir(exist_ok=True)
    SESSION_FILE.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    print(f"\n[saved session -> {SESSION_FILE.relative_to(PROJECT_ROOT)}]")


def load_session() -> AgentSession | None:
    """Load the session from disk, or return None if there is none yet."""
    if not SESSION_FILE.exists():
        return None
    return AgentSession.from_dict(json.loads(SESSION_FILE.read_text(encoding="utf-8")))


async def main() -> None:
    print_header(
        "12 - Session Persistence",
        "First run: start a trip plan and save it. Second run: load it and continue.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="TripPlanner",
        instructions="You are a travel planner. Keep answers short (max 4 bullet points).",
    )

    session = load_session()
    if session is None:
        print("No saved session found -> starting a NEW conversation.")
        session = agent.create_session()
        message = "I want to plan a 3-day trip to Kyoto in April. I love temples and food."
    else:
        print(f"Found saved session {session.session_id} -> RESUMING the conversation.")
        message = "Based on what I told you earlier, what should I do on day 2? Also remind me where I'm going."

    print_agent("User", message)
    response = await agent.run(message, session=session)
    print_agent(agent.name or "Agent", response.text)

    save_session(session)
    print("Run this example again to continue the same conversation. Delete the file to start over.")


if __name__ == "__main__":
    asyncio.run(main())
