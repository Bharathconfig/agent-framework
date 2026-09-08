"""This module defines functionality for packages/redis/agent_framework_redis/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Redis-owned feature-usage indexes."""

    REDIS = 68
