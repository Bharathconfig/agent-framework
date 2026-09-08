"""This module defines functionality for packages/mistral/agent_framework_mistral/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._chat_client import MistralChatClient, MistralChatOptions, MistralSettings, RawMistralChatClient
from ._embedding_client import MistralEmbeddingClient, MistralEmbeddingOptions, MistralEmbeddingSettings

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "MistralChatClient",
    "MistralChatOptions",
    "MistralEmbeddingClient",
    "MistralEmbeddingOptions",
    "MistralEmbeddingSettings",
    "MistralSettings",
    "RawMistralChatClient",
    "__version__",
]
