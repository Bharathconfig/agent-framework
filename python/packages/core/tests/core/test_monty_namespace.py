"""This module defines functionality for packages/core/tests/core/test_monty_namespace.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import sys
from types import ModuleType

import pytest

import agent_framework.monty as monty


def test_monty_namespace_dir_lists_lazy_exports() -> None:
    """Validates behavior for monty namespace dir lists lazy exports.
    """
    names = dir(monty)
    for expected in (
        "FileMount",
        "FileMountInput",
        "MontyCodeActProvider",
        "MontyExecuteCodeTool",
        "MountMode",
    ):
        assert expected in names


def test_monty_namespace_lazy_loads_known_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for monty namespace lazy loads known attribute.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    sentinel = object()
    fake_module = ModuleType("agent_framework_monty")
    fake_module.MontyCodeActProvider = sentinel  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "agent_framework_monty", fake_module)

    assert monty.MontyCodeActProvider is sentinel


def test_monty_namespace_unknown_attribute_raises_attribute_error() -> None:
    """Validates behavior for monty namespace unknown attribute raises attribute error.
    """
    with pytest.raises(AttributeError, match="Module `monty` has no attribute DoesNotExist."):
        _ = monty.DoesNotExist  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


def test_monty_namespace_missing_package_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for monty namespace missing package raises helpful error.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    monkeypatch.setitem(sys.modules, "agent_framework_monty", None)

    with pytest.raises(ModuleNotFoundError, match="agent-framework-monty"):
        _ = monty.MontyCodeActProvider
