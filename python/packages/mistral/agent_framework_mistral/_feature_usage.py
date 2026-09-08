"""This module defines functionality for packages/mistral/agent_framework_mistral/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Mistral-owned feature-usage indexes."""

    MISTRAL = 60
