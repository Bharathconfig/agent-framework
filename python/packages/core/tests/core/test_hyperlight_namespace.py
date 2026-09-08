"""This module defines functionality for packages/core/tests/core/test_hyperlight_namespace.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import sys
from types import ModuleType

import pytest

import agent_framework.hyperlight as hyperlight


def test_hyperlight_namespace_dir_lists_lazy_exports() -> None:
    """Validates behavior for hyperlight namespace dir lists lazy exports.
    """
    names = dir(hyperlight)
    for expected in (
        "AllowedDomain",
        "AllowedDomainInput",
        "FileMount",
        "FileMountInput",
        "HyperlightCodeActProvider",
        "HyperlightExecuteCodeTool",
    ):
        assert expected in names


def test_hyperlight_namespace_lazy_loads_known_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for hyperlight namespace lazy loads known attribute.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    sentinel = object()
    fake_module = ModuleType("agent_framework_hyperlight")
    fake_module.HyperlightCodeActProvider = sentinel  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "agent_framework_hyperlight", fake_module)

    assert hyperlight.HyperlightCodeActProvider is sentinel


def test_hyperlight_namespace_unknown_attribute_raises_attribute_error() -> None:
    """Validates behavior for hyperlight namespace unknown attribute raises attribute error.
    """
    with pytest.raises(AttributeError, match="Module `hyperlight` has no attribute DoesNotExist."):
        _ = hyperlight.DoesNotExist  # type: ignore[attr-defined]


def test_hyperlight_namespace_missing_package_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for hyperlight namespace missing package raises helpful error.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    monkeypatch.setitem(sys.modules, "agent_framework_hyperlight", None)

    with pytest.raises(ModuleNotFoundError, match="agent-framework-hyperlight"):
        _ = hyperlight.HyperlightCodeActProvider
