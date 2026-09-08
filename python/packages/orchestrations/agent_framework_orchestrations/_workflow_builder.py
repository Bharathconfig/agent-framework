"""This module defines functionality for packages/orchestrations/agent_framework_orchestrations/_workflow_builder.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from typing import ClassVar

from agent_framework._telemetry import FeatureIndex as CoreFeatureIndex
from agent_framework._workflows._workflow_builder import WorkflowBuilder


class OrchestrationWorkflowBuilder(WorkflowBuilder):
    """Workflow builder that leaves usage attribution to the orchestration."""

    _FEATURE_USAGE_INDEX: ClassVar[CoreFeatureIndex | None] = None
