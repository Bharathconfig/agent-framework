"""This module defines functionality for packages/redis/agent_framework_redis/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.
import importlib.metadata

from ._context_provider import RedisContextProvider
from ._history_provider import RedisHistoryProvider

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "RedisContextProvider",
    "RedisHistoryProvider",
    "__version__",
]
