"""This module defines functionality for packages/gemini/agent_framework_gemini/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._chat_client import (
    GeminiChatClient,
    GeminiChatOptions,
    GeminiSettings,
    GoogleGeminiSettings,
    RawGeminiChatClient,
    ThinkingConfig,
)

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "GeminiChatClient",
    "GeminiChatOptions",
    "GeminiSettings",
    "GoogleGeminiSettings",
    "RawGeminiChatClient",
    "ThinkingConfig",
    "__version__",
]
