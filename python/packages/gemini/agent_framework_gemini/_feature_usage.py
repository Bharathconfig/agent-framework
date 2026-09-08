"""This module defines functionality for packages/gemini/agent_framework_gemini/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Gemini-owned feature-usage indexes."""

    GEMINI = 59
