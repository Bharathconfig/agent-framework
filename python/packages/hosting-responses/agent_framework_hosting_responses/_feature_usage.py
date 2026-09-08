"""This module defines functionality for packages/hosting-responses/agent_framework_hosting_responses/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """OpenAI Responses hosting-owned feature-usage indexes."""

    HOSTING_RESPONSES = 86
