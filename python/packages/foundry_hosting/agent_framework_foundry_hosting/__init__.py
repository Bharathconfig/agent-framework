"""This module defines functionality for packages/foundry_hosting/agent_framework_foundry_hosting/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._invocations import InvocationsHostServer
from ._responses import ResponsesHostServer
from ._state_store import (
    AgentSessionStoreProvider,
    CheckpointStoreProvider,
    ContextScopedStoreProvider,
    FoundryAgentSessionStore,
    FoundryCheckpointStore,
    FoundryFunctionApprovalStore,
    FunctionApprovalStore,
    FunctionApprovalStoreProvider,
    StoreProvider,
)
from ._toolbox import FoundryToolbox

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AgentSessionStoreProvider",
    "CheckpointStoreProvider",
    "ContextScopedStoreProvider",
    "FoundryAgentSessionStore",
    "FoundryCheckpointStore",
    "FoundryFunctionApprovalStore",
    "FoundryToolbox",
    "FunctionApprovalStore",
    "FunctionApprovalStoreProvider",
    "InvocationsHostServer",
    "ResponsesHostServer",
    "StoreProvider",
]
