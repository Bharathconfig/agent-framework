"""This module defines functionality for packages/ollama/agent_framework_ollama/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Ollama-owned feature-usage indexes."""

    OLLAMA = 61
