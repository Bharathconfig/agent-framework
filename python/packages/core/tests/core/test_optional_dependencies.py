"""This module defines functionality for packages/core/tests/core/test_optional_dependencies.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys

import pytest

import agent_framework
import agent_framework.observability as observability
from agent_framework import Agent


def _hide_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Implements  hide otel sdk.
    
    Args:
        monkeypatch: Description of monkeypatch.
    
    Returns:
        Description of the return value.
    """
    import builtins

    real_import = builtins.__import__
    for module_name in list(sys.modules):
        if module_name == "opentelemetry.sdk" or module_name.startswith("opentelemetry.sdk."):
            sys.modules.pop(module_name, None)

    def _import_without_otel_sdk(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Implements  import without otel sdk.
        
        Args:
            name: Description of name.
            globals_: Description of globals_.
            locals_: Description of locals_.
            fromlist: Description of fromlist.
            level: Description of level.
        
        Returns:
            Description of the return value.
        """
        if name == "opentelemetry.sdk" or name.startswith("opentelemetry.sdk."):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_otel_sdk)


def test_create_resource_requires_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for create resource requires otel sdk.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    _hide_otel_sdk(monkeypatch)

    with pytest.raises(ModuleNotFoundError, match="opentelemetry-sdk"):
        observability.create_resource()


def test_observability_settings_initializes_without_cached_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for observability settings initializes without cached resource.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    _hide_otel_sdk(monkeypatch)

    settings = observability.ObservabilitySettings()

    assert not hasattr(settings, "_resource")


def test_configure_otel_providers_requires_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for configure otel providers requires otel sdk.
    
    Args:
        monkeypatch: Description of monkeypatch.
    """
    _hide_otel_sdk(monkeypatch)
    for key in [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "VS_CODE_EXTENSION_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ModuleNotFoundError, match="opentelemetry-sdk"):
        observability.configure_otel_providers()


def test_agent_framework_mcp_exports_remain_importable_without_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for agent framework mcp exports remain importable without mcp.
    
    Args:
        monkeypatch: Description of monkeypatch.
    
    Returns:
        Description of the return value.
    """
    import builtins

    import agent_framework._mcp as mcp_module

    real_import = builtins.__import__

    def _import_without_mcp(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Implements  import without mcp.
        
        Args:
            name: Description of name.
            globals_: Description of globals_.
            locals_: Description of locals_.
            fromlist: Description of fromlist.
            level: Description of level.
        
        Returns:
            Description of the return value.
        """
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError("No module named 'mcp'")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_mcp)
    assert agent_framework.MCPStdioTool is mcp_module.MCPStdioTool

    with pytest.raises(ModuleNotFoundError, match=r"Please install `mcp`\.$"):
        agent_framework.MCPStdioTool(name="test", command="python").get_mcp_client()


def test_mcp_streamable_http_tool_requires_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for mcp streamable http tool requires mcp.
    
    Args:
        monkeypatch: Description of monkeypatch.
    
    Returns:
        Description of the return value.
    """
    import builtins

    real_import = builtins.__import__

    def _import_without_mcp(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Implements  import without mcp.
        
        Args:
            name: Description of name.
            globals_: Description of globals_.
            locals_: Description of locals_.
            fromlist: Description of fromlist.
            level: Description of level.
        
        Returns:
            Description of the return value.
        """
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError("No module named 'mcp'")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_mcp)

    with pytest.raises(ModuleNotFoundError, match=r"Please install `mcp`\.$"):
        agent_framework.MCPStreamableHTTPTool(name="test", url="https://example.com").get_mcp_client()


def test_agent_as_mcp_server_requires_mcp(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for agent as mcp server requires mcp.
    
    Args:
        client: Description of client.
        monkeypatch: Description of monkeypatch.
    
    Returns:
        Description of the return value.
    """
    import builtins

    real_import = builtins.__import__

    def _import_without_mcp(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Implements  import without mcp.
        
        Args:
            name: Description of name.
            globals_: Description of globals_.
            locals_: Description of locals_.
            fromlist: Description of fromlist.
            level: Description of level.
        
        Returns:
            Description of the return value.
        """
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError("No module named 'mcp'")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_mcp)

    agent = Agent(client=client)  # type: ignore[arg-type]

    with pytest.raises(ModuleNotFoundError, match=r"Please install `mcp`\.$"):
        agent.as_mcp_server()


def test_mcp_websocket_tool_requires_ws_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for mcp websocket tool requires ws support.
    
    Args:
        monkeypatch: Description of monkeypatch.
    
    Returns:
        Description of the return value.
    """
    import builtins

    real_import = builtins.__import__

    sys.modules.pop("mcp.client.websocket", None)

    def _import_without_websocket_support(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Implements  import without websocket support.
        
        Args:
            name: Description of name.
            globals_: Description of globals_.
            locals_: Description of locals_.
            fromlist: Description of fromlist.
            level: Description of level.
        
        Returns:
            Description of the return value.
        """
        if name == "mcp.client.websocket":
            raise ModuleNotFoundError("No module named 'websockets'", name="websockets")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_websocket_support)

    with pytest.raises(ModuleNotFoundError, match=r"mcp\[ws\]"):
        agent_framework.MCPWebsocketTool(name="test", url="wss://example.com").get_mcp_client()


def test_mcp_websocket_tool_requires_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates behavior for mcp websocket tool requires mcp.
    
    Args:
        monkeypatch: Description of monkeypatch.
    
    Returns:
        Description of the return value.
    """
    import builtins

    real_import = builtins.__import__
    sys.modules.pop("mcp.client.websocket", None)

    def _import_without_mcp(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """Implements  import without mcp.
        
        Args:
            name: Description of name.
            globals_: Description of globals_.
            locals_: Description of locals_.
            fromlist: Description of fromlist.
            level: Description of level.
        
        Returns:
            Description of the return value.
        """
        if name == "mcp.client.websocket":
            raise ModuleNotFoundError("No module named 'mcp.client.websocket'", name="mcp.client.websocket")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_mcp)

    with pytest.raises(ModuleNotFoundError, match=r"agent-framework-core\[mcp\]|mcp\[ws\]"):
        agent_framework.MCPWebsocketTool(name="test", url="wss://example.com").get_mcp_client()
