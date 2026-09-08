"""This module defines functionality for packages/core/tests/core/test_tools_namespace.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import sys
from types import ModuleType

import pytest

import agent_framework.tools as tools


def test_tools_namespace_dir_lists_lazy_exports() -> None:
    """Validates behavior for tools namespace dir lists lazy exports.
    """
    names = dir(tools)
    for expected in (
        "DockerShellTool",
        "LocalShellTool",
        "ShellEnvironmentProvider",
        "ShellEnvironmentProviderOptions",
        "ShellExecutor",
        "ShellPolicy",
    ):
        assert expected in names


def test_tools_namespace_lazy_loads_known_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for tools namespace lazy loads known attribute.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    sentinel = object()
    fake_module = ModuleType("agent_framework_tools.shell")
    fake_module.LocalShellTool = sentinel  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "agent_framework_tools.shell", fake_module)

    assert tools.LocalShellTool is sentinel


def test_tools_namespace_unknown_attribute_raises_attribute_error() -> None:
    """Validates behavior for tools namespace unknown attribute raises attribute error.
    """
    with pytest.raises(AttributeError, match="Module `tools` has no attribute DoesNotExist."):
        _ = tools.DoesNotExist  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


def test_tools_namespace_missing_package_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for tools namespace missing package raises helpful error.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    monkeypatch.setitem(sys.modules, "agent_framework_tools.shell", None)

    with pytest.raises(ModuleNotFoundError, match="agent-framework-tools"):
        _ = tools.LocalShellTool
