"""This module defines functionality for packages/claude/agent_framework_claude/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Claude Agent SDK-owned feature-usage indexes."""

    CLAUDE = 62
