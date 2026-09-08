"""This module defines functionality for packages/azure-cosmos/agent_framework_azure_cosmos/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Azure Cosmos DB-owned feature-usage indexes."""

    AZURE_COSMOS = 66
