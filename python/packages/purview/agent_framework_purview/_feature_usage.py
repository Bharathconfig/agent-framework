"""This module defines functionality for packages/purview/agent_framework_purview/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Purview-owned feature-usage indexes."""

    PURVIEW = 70
