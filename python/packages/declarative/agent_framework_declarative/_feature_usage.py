"""This module defines functionality for packages/declarative/agent_framework_declarative/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Declarative-owned feature-usage indexes."""

    DECLARATIVE_AGENT = 75
    DECLARATIVE_WORKFLOW = 76
