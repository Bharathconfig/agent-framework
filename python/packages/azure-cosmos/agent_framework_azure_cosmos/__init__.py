"""This module defines functionality for packages/azure-cosmos/agent_framework_azure_cosmos/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._checkpoint_storage import CosmosCheckpointStorage
from ._history_provider import CosmosHistoryProvider

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "CosmosCheckpointStorage",
    "CosmosHistoryProvider",
    "__version__",
]
