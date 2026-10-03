"""
common/config.py
================

Central configuration for EVERY example in this repository.

Why this file exists
--------------------
All 25 examples need exactly the same thing: an authenticated chat client that
talks to your Azure OpenAI deployment. Instead of repeating the connection code
25 times, every example simply calls :func:`get_chat_client` from here.

That means the ONLY thing you ever have to do is fill in your ``.env`` file:

    AZURE_OPENAI_API_KEY=<your key>
    AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/

Everything else (deployment name, API version) has a sensible default that you
can optionally override in ``.env``.

How it works
------------
1. ``python-dotenv`` loads the ``.env`` file that lives in the project root
   (next to ``requirements.txt``) into ``os.environ``.
2. :func:`load_settings` reads the values, applies defaults and validates that
   the two required values are present. If something is missing you get a
   friendly, actionable error instead of a stack trace from deep inside the SDK.
3. :func:`get_chat_client` builds an ``OpenAIChatCompletionClient`` from the
   Microsoft Agent Framework, pointed at your Azure OpenAI resource.

The Chat Completions API is used (instead of the newer Responses API) because
it is supported by every Azure OpenAI model deployment and API version, which
keeps the examples working out-of-the-box for the widest range of setups.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Locate and load the .env file
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this "common" package, i.e. the
# repository root. We load ".env" from there so that examples work no matter
# which directory you launch them from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"

# override=False -> real environment variables (e.g. set in CI or your shell)
# always win over values in the .env file.
load_dotenv(ENV_FILE, override=False)

# ---------------------------------------------------------------------------
# 2. Defaults for the optional settings
# ---------------------------------------------------------------------------
# The *deployment name* you chose when deploying a model in Azure AI Foundry /
# Azure OpenAI Studio. "gpt-4o-mini" is cheap, fast and supports tools +
# structured outputs, so it is a great default for learning.
DEFAULT_DEPLOYMENT = "gpt-4o-mini"

# A GA (generally available) Azure OpenAI API version that supports tool
# calling and JSON-schema structured outputs.
DEFAULT_API_VERSION = "2024-10-21"

# Placeholder values shipped in .env.example. If we still see them, the user
# has not configured their credentials yet.
_PLACEHOLDERS = {"", "<your-api-key>", "https://<your-resource-name>.openai.azure.com/"}


@dataclass(frozen=True)
class Settings:
    """Strongly-typed bundle of the connection settings."""

    api_key: str
    endpoint: str
    deployment: str
    api_version: str


def load_settings() -> Settings:
    """Read, default and validate the Azure OpenAI settings from the environment.

    Returns:
        A :class:`Settings` instance.

    Exits:
        With a helpful message (exit code 1) if the API key or endpoint is missing.
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip() or DEFAULT_DEPLOYMENT
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip() or DEFAULT_API_VERSION

    missing = [
        name
        for name, value in (("AZURE_OPENAI_API_KEY", api_key), ("AZURE_OPENAI_ENDPOINT", endpoint))
        if value in _PLACEHOLDERS
    ]
    if missing:
        # We deliberately exit instead of raising: beginners get a clean,
        # readable instruction rather than a long traceback.
        sys.exit(
            "\n[configuration error] Missing: "
            + ", ".join(missing)
            + f"\n  1. Copy '.env.example' to '.env' in {PROJECT_ROOT}"
            + "\n  2. Paste your Azure OpenAI API key and endpoint into it"
            + "\n  3. Run the example again.\n"
        )

    return Settings(api_key=api_key, endpoint=endpoint, deployment=deployment, api_version=api_version)


def get_chat_client():
    """Create the chat client shared by all examples.

    The returned object is an ``agent_framework.openai.OpenAIChatCompletionClient``
    configured for Azure OpenAI. You pass it to ``agent_framework.Agent(client=...)``.

    A new client is cheap to create, but you can (and in multi-agent examples we
    do) share one client between many agents.
    """
    # Imported lazily so that "python run_example.py --list" works even before
    # the dependencies are installed.
    from agent_framework.openai import OpenAIChatCompletionClient

    settings = load_settings()
    return OpenAIChatCompletionClient(
        model=settings.deployment,  # For Azure this is your *deployment name*
        azure_endpoint=settings.endpoint,  # https://<resource>.openai.azure.com/
        api_key=settings.api_key,  # Key auth (simplest way to get started)
        api_version=settings.api_version,
    )
