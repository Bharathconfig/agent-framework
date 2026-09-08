# Copyright (c) Microsoft. All rights reserved.

"""Tests for ``InvokeMcpToolActionExecutor``.

Use a stub :class:`MCPToolHandler` that returns canned :class:`MCPToolResult`s.
No real MCP server or network is exercised. See
``test_default_mcp_tool_handler.py`` for tests that exercise the real
``DefaultMCPToolHandler`` against a mocked ``MCPStreamableHTTPTool``.
"""

import sys
from typing import Any

import httpx
import pytest

try:
    import powerfx  # noqa: F401

    _powerfx_available = True
except (ImportError, RuntimeError):
    _powerfx_available = False

pytestmark = pytest.mark.skipif(
    not _powerfx_available or sys.version_info >= (3, 14),
    reason="PowerFx engine not available (requires dotnet runtime)",
)

from agent_framework import Content, Message  # noqa: E402
from agent_framework.exceptions import ToolExecutionException  # noqa: E402

from agent_framework_declarative._workflows import (  # noqa: E402
    DECLARATIVE_STATE_KEY,
    DeclarativeWorkflowError,
    MCPToolHandler,
    MCPToolInvocation,
    MCPToolResult,
    WorkflowFactory,
)


class StubMcpHandler:
    """Test stub recording the last call and returning a canned result."""

    def __init__(
        self,
        result: MCPToolResult | None = None,
        *,
        raise_exc: BaseException | None = None,
    ) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
            result: Description of result.
            raise_exc: Description of raise_exc.
        """
        self.result = result
        self.raise_exc = raise_exc
        self.last_invocation: MCPToolInvocation | None = None
        self.invocations: list[MCPToolInvocation] = []
        self.call_count = 0

    async def invoke_tool(self, invocation: MCPToolInvocation) -> MCPToolResult:
        """Implements invoke tool.
        
        Args:
            self: Description of self.
            invocation: Description of invocation.
        
        Returns:
            Description of the return value.
        """
        self.call_count += 1
        self.last_invocation = invocation
        self.invocations.append(invocation)
        if self.raise_exc is not None:
            raise self.raise_exc
        assert self.result is not None
        return self.result


def _ok(outputs: list[Content] | None = None) -> MCPToolResult:
    """Implements  ok.
    
    Args:
        outputs: Description of outputs.
    
    Returns:
        Description of the return value.
    """
    return MCPToolResult(outputs=outputs or [Content.from_text("hello")])


def _err(message: str = "boom") -> MCPToolResult:
    """Implements  err.
    
    Args:
        message: Description of message.
    
    Returns:
        Description of the return value.
    """
    return MCPToolResult(
        outputs=[Content.from_text(f"Error: {message}")],
        is_error=True,
        error_message=message,
    )


def _action(
    *,
    server_url: str = "https://mcp.example/api",
    tool_name: str = "search",
    server_label: str | None = None,
    arguments: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    require_approval: Any = None,
    connection: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Implements  action.
    
    Args:
        server_url: Description of server_url.
        tool_name: Description of tool_name.
        server_label: Description of server_label.
        arguments: Description of arguments.
        headers: Description of headers.
        require_approval: Description of require_approval.
        connection: Description of connection.
        conversation_id: Description of conversation_id.
        output: Description of output.
    
    Returns:
        Description of the return value.
    """
    action: dict[str, Any] = {
        "kind": "InvokeMcpTool",
        "id": "mcp_action",
        "serverUrl": server_url,
        "toolName": tool_name,
    }
    if server_label is not None:
        action["serverLabel"] = server_label
    if arguments is not None:
        action["arguments"] = arguments
    if headers is not None:
        action["headers"] = headers
    if require_approval is not None:
        action["requireApproval"] = require_approval
    if connection is not None:
        action["connection"] = connection
    if conversation_id is not None:
        action["conversationId"] = conversation_id
    if output is not None:
        action["output"] = output
    return action


def _yaml(action: dict[str, Any]) -> dict[str, Any]:
    """Implements  yaml.
    
    Args:
        action: Description of action.
    
    Returns:
        Description of the return value.
    """
    return {"name": "mcp_test", "actions": [action]}


# ---------- Builder enforcement --------------------------------------------


class TestBuilderEnforcement:
    """Represents the TestBuilderEnforcement type and related behavior.
    """
    def test_missing_handler_raises_at_build_time(self) -> None:
        """Validates behavior for missing handler raises at build time.
        
        Args:
            self: Description of self.
        """
        factory = WorkflowFactory()
        with pytest.raises(DeclarativeWorkflowError) as excinfo:
            factory.create_workflow_from_definition(_yaml(_action()))
        assert "InvokeMcpTool" in str(excinfo.value)
        assert "mcp_tool_handler" in str(excinfo.value)

    def test_missing_server_url_fails_validation(self) -> None:
        """Validates behavior for missing server url fails validation.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        factory = WorkflowFactory(mcp_tool_handler=handler)
        action = _action()
        del action["serverUrl"]
        with pytest.raises(Exception) as excinfo:
            factory.create_workflow_from_definition(_yaml(action))
        assert "serverUrl" in str(excinfo.value)

    def test_missing_tool_name_fails_validation(self) -> None:
        """Validates behavior for missing tool name fails validation.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        factory = WorkflowFactory(mcp_tool_handler=handler)
        action = _action()
        del action["toolName"]
        with pytest.raises(Exception) as excinfo:
            factory.create_workflow_from_definition(_yaml(action))
        assert "toolName" in str(excinfo.value)


# ---------- Field forwarding ----------------------------------------------


class TestFieldForwarding:
    """Represents the TestFieldForwarding type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_basic_invocation_forwards_required_fields(self) -> None:
        """Validates behavior for basic invocation forwards required fields.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action()))
        await workflow.run({})
        assert handler.call_count == 1
        inv = handler.last_invocation
        assert inv is not None
        assert inv.server_url == "https://mcp.example/api"
        assert inv.tool_name == "search"
        assert inv.server_label is None
        assert inv.headers == {}
        assert inv.arguments == {}
        assert inv.connection_name is None

    @pytest.mark.asyncio
    async def test_arguments_evaluated_and_preserves_none(self) -> None:
        """Validates behavior for arguments evaluated and preserves none.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(
            _yaml(
                _action(
                    arguments={
                        "query": "weather today",
                        "limit": 5,
                        "fresh": True,
                        "missing": None,
                    }
                )
            )
        )
        await workflow.run({})
        inv = handler.last_invocation
        assert inv is not None
        # ``None`` is preserved (parity with .NET) — caller decides.
        assert inv.arguments == {
            "query": "weather today",
            "limit": 5,
            "fresh": True,
            "missing": None,
        }

    @pytest.mark.asyncio
    async def test_headers_drop_empty_values(self) -> None:
        """Validates behavior for headers drop empty values.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(
            _yaml(
                _action(
                    headers={
                        "Authorization": "Bearer token-123",
                        "X-Trace": "trace-id",
                        "X-Empty": "",
                    }
                )
            )
        )
        await workflow.run({})
        inv = handler.last_invocation
        assert inv is not None
        assert inv.headers == {
            "Authorization": "Bearer token-123",
            "X-Trace": "trace-id",
        }

    @pytest.mark.asyncio
    async def test_server_label_and_connection_name_forwarded(self) -> None:
        """Validates behavior for server label and connection name forwarded.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(
            _yaml(
                _action(
                    server_label="docs-mcp",
                    connection={"name": "azure-conn"},
                )
            )
        )
        await workflow.run({})
        inv = handler.last_invocation
        assert inv is not None
        assert inv.server_label == "docs-mcp"
        assert inv.connection_name == "azure-conn"


# ---------- Output handling ------------------------------------------------


class TestOutput:
    """Represents the TestOutput type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_output_result_parses_json_text(self) -> None:
        """Validates behavior for output result parses json text.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text('{"k":"v","n":1}')]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": "Local.Result"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        assert decl["Local"]["Result"] == [{"k": "v", "n": 1}]

    @pytest.mark.asyncio
    async def test_output_result_falls_back_to_raw_text(self) -> None:
        """Validates behavior for output result falls back to raw text.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("plain text not json")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": "Local.Result"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        assert decl["Local"]["Result"] == ["plain text not json"]

    @pytest.mark.asyncio
    async def test_output_messages_writes_single_tool_role_message(self) -> None:
        """Validates behavior for output messages writes single tool role message.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("hi"), Content.from_text("there")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"messages": "Local.Messages"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        msg = decl["Local"]["Messages"]
        # Single Tool-role message containing both contents (parity with .NET).
        assert isinstance(msg, Message)
        assert str(msg.role).lower() == "tool"
        assert len(msg.contents) == 2

    @pytest.mark.asyncio
    async def test_uri_content_serialised_as_uri_string(self) -> None:
        """Validates behavior for uri content serialised as uri string.
        
        Args:
            self: Description of self.
        """
        uri_content = Content.from_uri("https://example.com/file.txt", media_type="text/plain")
        handler = StubMcpHandler(_ok([uri_content]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": "Local.Result"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        assert decl["Local"]["Result"] == ["https://example.com/file.txt"]

    @pytest.mark.asyncio
    async def test_output_path_object_form(self) -> None:
        """Validates behavior for output path object form.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("ok")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": {"path": "Local.Result"}})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        assert decl["Local"]["Result"] == ["ok"]


# ---------- Conversation append --------------------------------------------


class TestConversation:
    """Represents the TestConversation type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_conversation_id_appends_assistant_message(self) -> None:
        """Validates behavior for conversation id appends assistant message.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("answer")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(
            _yaml(
                _action(
                    conversation_id="conv-42",
                    output={"result": "Local.Result"},
                )
            )
        )
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        conv = decl["System"]["conversations"]["conv-42"]
        msgs = conv["messages"] if isinstance(conv, dict) else conv.messages
        assert len(msgs) == 1
        appended = msgs[0]
        assert str(appended.role).lower() == "assistant"
        # Same contents as the tool output.
        assert len(appended.contents) == 1

    @pytest.mark.asyncio
    async def test_empty_conversation_id_does_not_append(self) -> None:
        """Validates behavior for empty conversation id does not append.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("answer")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(
            _yaml(
                _action(
                    conversation_id="",
                    output={"result": "Local.Result"},
                )
            )
        )
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        # Empty conversation id must not produce a `""` entry under System.conversations.
        conversations = decl.get("System", {}).get("conversations", {})
        assert "" not in conversations


# ---------- Approval flow --------------------------------------------------


@pytest.fixture
def mock_state():  # type: ignore[no-untyped-def]
    """Implements mock state.
    
    Returns:
        Description of the return value.
    """
    from unittest.mock import MagicMock

    state = MagicMock()
    state._data = {}

    def _get(key: str, default: Any = None) -> Any:
        """Implements  get.
        
        Args:
            key: Description of key.
            default: Description of default.
        
        Returns:
            Description of the return value.
        """
        if key not in state._data:
            if default is not None:
                return default
            raise KeyError(key)
        return state._data[key]

    def _set(key: str, value: Any) -> None:
        """Implements  set.
        
        Args:
            key: Description of key.
            value: Description of value.
        """
        state._data[key] = value

    def _delete(key: str) -> None:
        """Implements  delete.
        
        Args:
            key: Description of key.
        """
        if key in state._data:
            del state._data[key]
        else:
            raise KeyError(key)

    state.get = MagicMock(side_effect=_get)
    state.set = MagicMock(side_effect=_set)
    state.delete = MagicMock(side_effect=_delete)
    return state


@pytest.fixture
def mock_context(mock_state):  # type: ignore[no-untyped-def]
    """Implements mock context.
    
    Args:
        mock_state: Description of mock_state.
    
    Returns:
        Description of the return value.
    """
    from unittest.mock import AsyncMock, MagicMock

    ctx = MagicMock()
    ctx.state = mock_state
    ctx.send_message = AsyncMock()
    ctx.yield_output = AsyncMock()
    ctx.request_info = AsyncMock()
    return ctx


def _seed_state(mock_state) -> None:  # type: ignore[no-untyped-def]
    """Pre-seed the declarative state container as the executors expect."""
    from agent_framework_declarative._workflows import DECLARATIVE_STATE_KEY

    mock_state._data[DECLARATIVE_STATE_KEY] = {
        "Local": {},
        "Custom": {},
        "Workflow": {},
        "System": {
            "ConversationId": "00000000-0000-0000-0000-000000000000",
            "LastMessage": {"Id": "", "Text": ""},
            "LastMessageText": "",
            "LastMessageId": "",
        },
        "Agent": {},
        "Conversation": {"messages": [], "history": []},
        "Inputs": {},
    }


class TestApprovalFlow:
    """Represents the TestApprovalFlow type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_approval_required_emits_request_and_yields(self, mock_state, mock_context) -> None:  # type: ignore[no-untyped-def]
        """Validates behavior for approval required emits request and yields.
        
        Args:
            self: Description of self.
            mock_state: Description of mock_state.
            mock_context: Description of mock_context.
        """
        from agent_framework_declarative._workflows._declarative_base import ActionTrigger
        from agent_framework_declarative._workflows._executors_mcp import (
            InvokeMcpToolActionExecutor,
            MCPToolApprovalRequest,
        )

        _seed_state(mock_state)
        handler = StubMcpHandler(_ok())
        executor = InvokeMcpToolActionExecutor(
            _action(
                require_approval=True,
                arguments={"q": "x"},
                headers={"Authorization": "Bearer SECRET"},
                output={"result": "Local.Result"},
            ),
            mcp_tool_handler=handler,
        )
        await executor.handle_action(ActionTrigger(), mock_context)

        # Approval request emitted.
        mock_context.request_info.assert_called_once()
        request = mock_context.request_info.call_args[0][0]
        assert isinstance(request, MCPToolApprovalRequest)
        assert request.tool_name == "search"
        assert request.arguments == {"q": "x"}
        assert request.header_names == ["Authorization"]

        # NEVER expose the actual auth token in any field of the approval payload.
        for value in request.__dict__.values():
            assert "SECRET" not in str(value)

        # Workflow should yield (no ActionComplete sent yet).
        mock_context.send_message.assert_not_called()

        # Handler not invoked yet.
        assert handler.call_count == 0

    @pytest.mark.asyncio
    async def test_approval_response_approved_invokes_handler(self, mock_state, mock_context) -> None:  # type: ignore[no-untyped-def]
        """Validates behavior for approval response approved invokes handler.
        
        Args:
            self: Description of self.
            mock_state: Description of mock_state.
            mock_context: Description of mock_context.
        """
        from agent_framework_declarative._workflows import ActionComplete, ToolApprovalResponse
        from agent_framework_declarative._workflows._executors_mcp import (
            InvokeMcpToolActionExecutor,
            MCPToolApprovalRequest,
        )

        _seed_state(mock_state)
        handler = StubMcpHandler(_ok([Content.from_text('{"ok":true}')]))
        executor = InvokeMcpToolActionExecutor(
            _action(
                require_approval=True,
                headers={"Authorization": "Bearer tk"},
                output={"result": "Local.Result"},
            ),
            mcp_tool_handler=handler,
        )
        await executor.handle_approval_response(
            MCPToolApprovalRequest(
                request_id="req-1",
                tool_name="search",
                server_url="https://mcp.example/api",
                server_label=None,
                arguments={"q": "x"},
            ),
            ToolApprovalResponse(approved=True),
            mock_context,
        )

        assert handler.call_count == 1
        inv = handler.last_invocation
        assert inv is not None
        # Invocation fields source from the approval request payload.
        assert inv.tool_name == "search"
        assert inv.server_url == "https://mcp.example/api"
        assert inv.arguments == {"q": "x"}
        # Headers are re-evaluated from the action definition on resume.
        assert inv.headers == {"Authorization": "Bearer tk"}
        # ActionComplete was sent.
        mock_context.send_message.assert_called_once()
        sent = mock_context.send_message.call_args[0][0]
        assert isinstance(sent, ActionComplete)

    @pytest.mark.asyncio
    async def test_approval_response_rejected_assigns_error(self, mock_state, mock_context) -> None:  # type: ignore[no-untyped-def]
        """Validates behavior for approval response rejected assigns error.
        
        Args:
            self: Description of self.
            mock_state: Description of mock_state.
            mock_context: Description of mock_context.
        """
        from agent_framework_declarative._workflows import ToolApprovalResponse
        from agent_framework_declarative._workflows._executors_mcp import (
            InvokeMcpToolActionExecutor,
            MCPToolApprovalRequest,
        )

        _seed_state(mock_state)
        handler = StubMcpHandler(_ok())
        executor = InvokeMcpToolActionExecutor(
            _action(
                require_approval=True,
                output={"result": "Local.Result"},
            ),
            mcp_tool_handler=handler,
        )
        await executor.handle_approval_response(
            MCPToolApprovalRequest(
                request_id="req-2",
                tool_name="search",
                server_url="https://mcp.example/api",
                server_label=None,
                arguments={},
            ),
            ToolApprovalResponse(approved=False, reason="not authorized"),
            mock_context,
        )

        assert handler.call_count == 0
        # Error string assigned at output.result.
        from agent_framework_declarative._workflows import DECLARATIVE_STATE_KEY

        result = mock_state._data[DECLARATIVE_STATE_KEY]["Local"]["Result"]
        assert result == "Error: MCP tool invocation was not approved by user."


# ---------- Error handling -------------------------------------------------


class TestErrorHandling:
    """Represents the TestErrorHandling type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_handler_returns_error_result_assigns_error_string(self) -> None:
        """Validates behavior for handler returns error result assigns error string.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_err("server down"))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": "Local.Result"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        assert decl["Local"]["Result"] == "Error: server down"

    @pytest.mark.asyncio
    async def test_tool_execution_exception_becomes_error_result(self) -> None:
        """Validates behavior for tool execution exception becomes error result.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(raise_exc=ToolExecutionException("invalid arguments"))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": "Local.Result"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        assert decl["Local"]["Result"] == "Error: invalid arguments"

    @pytest.mark.asyncio
    async def test_httpx_error_becomes_error_result(self) -> None:
        """Validates behavior for httpx error becomes error result.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(raise_exc=httpx.ConnectError("dns fail"))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"result": "Local.Result"})))
        await workflow.run({})
        decl = workflow._runner.state.get(DECLARATIVE_STATE_KEY)
        result = decl["Local"]["Result"]
        assert isinstance(result, str)
        assert result.startswith("Error:")
        assert "ConnectError" in result

    @pytest.mark.asyncio
    async def test_unexpected_exception_propagates(self) -> None:
        """Programmer bugs (TypeError etc.) must NOT be swallowed."""
        handler = StubMcpHandler(raise_exc=TypeError("bad type"))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action()))
        with pytest.raises(Exception) as excinfo:
            await workflow.run({})
        # Either the TypeError reaches us or it gets wrapped by the runner —
        # either way the message must surface.
        assert "bad type" in str(excinfo.value)


# ---------- autoSend -------------------------------------------------------


class TestAutoSend:
    """Represents the TestAutoSend type and related behavior.
    """
    @pytest.mark.asyncio
    async def test_auto_send_default_true_yields_output(self) -> None:
        """Validates behavior for auto send default true yields output.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("hello")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action()))
        events = await workflow.run({})
        outputs = events.get_outputs()
        assert len(outputs) == 1

    @pytest.mark.asyncio
    async def test_auto_send_false_suppresses_yield(self) -> None:
        """Validates behavior for auto send false suppresses yield.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok([Content.from_text("hello")]))
        factory = WorkflowFactory(mcp_tool_handler=handler)
        workflow = factory.create_workflow_from_definition(_yaml(_action(output={"autoSend": False})))
        events = await workflow.run({})
        outputs = events.get_outputs()
        assert outputs == []


# ---------- Protocol structure --------------------------------------------


class TestProtocol:
    """Represents the TestProtocol type and related behavior.
    """
    def test_stub_handler_satisfies_protocol(self) -> None:
        """Validates behavior for stub handler satisfies protocol.
        
        Args:
            self: Description of self.
        """
        handler = StubMcpHandler(_ok())
        assert isinstance(handler, MCPToolHandler)


# ---------- _format_outputs_for_send --------------------------------------


class TestFormatOutputsForSend:
    """Direct tests for the auto-send rendering helper.

    Regression for PR #5630 review-comment 4: a single scalar JSON value
    must render bare (e.g. ``"42"``) rather than wrapped (``"[42]"``).
    """

    @pytest.mark.parametrize(
        ("parsed", "expected"),
        [
            ([], ""),
            (["hello"], "hello"),
            (["a", "b"], "a\nb"),
            ([42], "42"),
            ([3.14], "3.14"),
            ([True], "true"),
            ([False], "false"),
            ([None], "null"),
            ([{"k": "v"}], '{"k": "v"}'),
            ([[1, 2]], "[1, 2]"),
            (["hello", 42], '["hello", 42]'),
            ([{"a": 1}, {"b": 2}], '[{"a": 1}, {"b": 2}]'),
        ],
    )
    def test_format_outputs_for_send(self, parsed: list[Any], expected: str) -> None:
        """Validates behavior for format outputs for send.
        
        Args:
            self: Description of self.
            parsed: Description of parsed.
            expected: Description of expected.
        """
        from agent_framework_declarative._workflows._executors_mcp import _format_outputs_for_send

        assert _format_outputs_for_send(parsed) == expected
