"""This module defines functionality for packages/claude/agent_framework_claude/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._agent import ClaudeAgent, ClaudeAgentOptions, ClaudeAgentSettings, RawClaudeAgent

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "ClaudeAgent",
    "ClaudeAgentOptions",
    "ClaudeAgentSettings",
    "RawClaudeAgent",
    "__version__",
]
