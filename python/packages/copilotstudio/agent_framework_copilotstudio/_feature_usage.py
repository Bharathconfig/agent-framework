"""This module defines functionality for packages/copilotstudio/agent_framework_copilotstudio/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Copilot Studio-owned feature-usage indexes."""

    COPILOTSTUDIO = 63
