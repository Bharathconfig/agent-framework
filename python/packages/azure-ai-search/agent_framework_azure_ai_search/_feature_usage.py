"""This module defines functionality for packages/azure-ai-search/agent_framework_azure_ai_search/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Azure AI Search-owned feature-usage indexes."""

    AZURE_AI_SEARCH = 65
