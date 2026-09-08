"""This module defines functionality for packages/github_copilot/agent_framework_github_copilot/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """GitHub Copilot-owned feature-usage indexes."""

    GITHUB_COPILOT = 64
