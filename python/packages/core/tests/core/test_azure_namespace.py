"""This module defines functionality for packages/core/tests/core/test_azure_namespace.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import pytest

import agent_framework.azure as azure

CosmosHistoryProvider = pytest.importorskip("agent_framework_azure_cosmos").CosmosHistoryProvider


def test_azure_namespace_exposes_cosmos_history_provider() -> None:
    """Validates behavior for azure namespace exposes cosmos history provider.
    """
    assert azure.CosmosHistoryProvider is CosmosHistoryProvider
    assert "CosmosHistoryProvider" in dir(azure)
