"""This module defines functionality for packages/lab/namespace/agent_framework/lab/gaia/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

# Import and re-export from the actual implementation
from agent_framework_lab_gaia import (
    GAIA,
    Evaluation,
    Evaluator,
    GAIATelemetryConfig,
    Prediction,
    Task,
    TaskResult,
    TaskRunner,
    gaia_scorer,
    viewer_main,
)

__all__ = [
    "GAIA",
    "Evaluation",
    "Evaluator",
    "GAIATelemetryConfig",
    "Prediction",
    "Task",
    "TaskResult",
    "TaskRunner",
    "gaia_scorer",
    "viewer_main",
]
