"""This module defines functionality for packages/github_copilot/agent_framework_github_copilot/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._agent import GitHubCopilotAgent, GitHubCopilotOptions, GitHubCopilotSettings, RawGitHubCopilotAgent

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "GitHubCopilotAgent",
    "GitHubCopilotOptions",
    "GitHubCopilotSettings",
    "RawGitHubCopilotAgent",
    "__version__",
]
