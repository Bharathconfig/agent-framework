"""This module defines functionality for packages/core/tests/core/test_mistral_namespace.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import sys
from types import ModuleType

import pytest

import agent_framework.mistral as mistral


def test_mistral_namespace_dir_lists_lazy_exports() -> None:
    """Validates behavior for mistral namespace dir lists lazy exports.
    """
    names = dir(mistral)
    for expected in (
        "MistralChatClient",
        "MistralChatOptions",
        "MistralEmbeddingClient",
        "MistralEmbeddingOptions",
        "MistralEmbeddingSettings",
        "MistralSettings",
        "RawMistralChatClient",
    ):
        assert expected in names


def test_mistral_namespace_lazy_loads_known_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for mistral namespace lazy loads known attribute.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    sentinel = object()
    fake_module = ModuleType("agent_framework_mistral")
    fake_module.MistralEmbeddingClient = sentinel  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "agent_framework_mistral", fake_module)

    assert mistral.MistralEmbeddingClient is sentinel


def test_mistral_namespace_unknown_attribute_raises_attribute_error() -> None:
    """Validates behavior for mistral namespace unknown attribute raises attribute error.
    """
    with pytest.raises(AttributeError, match="Module `mistral` has no attribute DoesNotExist."):
        _ = mistral.DoesNotExist  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


def test_mistral_namespace_missing_package_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for mistral namespace missing package raises helpful error.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    monkeypatch.setitem(sys.modules, "agent_framework_mistral", None)

    with pytest.raises(ModuleNotFoundError, match="agent-framework-mistral"):
        _ = mistral.MistralEmbeddingClient
