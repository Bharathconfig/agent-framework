"""This module defines functionality for packages/bedrock/agent_framework_bedrock/_feature_usage.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class FeatureIndex(IntEnum):
    """Amazon Bedrock-owned feature-usage indexes."""

    BEDROCK = 58
