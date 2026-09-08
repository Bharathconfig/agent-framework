"""This module defines functionality for packages/foundry_hosting/agent_framework_foundry_hosting/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Foundry hosting-owned feature-usage indexes."""

    FOUNDRY_TOOLBOX = 53
    FOUNDRY_HOSTING = 55
