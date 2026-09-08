"""This module defines functionality for packages/anthropic/agent_framework_anthropic/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Anthropic-owned feature-usage indexes."""

    ANTHROPIC = 57
