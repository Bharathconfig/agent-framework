"""This module defines functionality for packages/azure-cosmos-memory/agent_framework_azure_cosmos_memory/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Azure Cosmos DB memory-owned feature-usage indexes."""

    AZURE_COSMOS_MEMORY = 82
