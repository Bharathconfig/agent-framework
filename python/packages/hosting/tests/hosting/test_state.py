"""This module defines functionality for packages/hosting/tests/hosting/test_state.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, Literal, overload

import pytest
from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    AgentSession,
    Content,
    Message,
    ResponseStream,
    SessionStore,
    Workflow,
)

import agent_framework_hosting
from agent_framework_hosting import AgentState, WorkflowState


def _workflow_fixture(name: str) -> Any:
    """Load a fixture from ``_workflow_fixtures.py`` via the ``conftest``-registered alias.

    Mirrors ``test_host.py``'s helper: the local ``conftest.py`` registers
    ``_workflow_fixtures.py`` under the collision-proof name
    ``hosting_workflow_fixtures`` so it stays importable in both
    package-local and aggregate pytest runs.
    """
    return getattr(importlib.import_module("hosting_workflow_fixtures"), name)


class _FakeAgent:
    """Minimal agent target for state tests.

    Declares ``run`` with the same two overloads as ``SupportsAgentRun`` (one
    per ``stream`` value) so it satisfies the protocol under static type
    checking, not just at runtime.
    """

    id: str = "fake-agent"
    name: str | None = "Fake Agent"
    description: str | None = "Fake agent for tests"

    def __init__(self) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
        """
        self.created_sessions: list[AgentSession] = []

    def create_session(self, *, session_id: str | None = None) -> AgentSession:
        """Implements create session.
        
        Args:
            self: Description of self.
            session_id: Description of session_id.
        
        Returns:
            Description of the return value.
        """
        session = AgentSession(session_id=session_id)
        self.created_sessions.append(session)
        return session

    def get_session(self, service_session_id: Any, *, session_id: str | None = None) -> AgentSession:
        """Implements get session.
        
        Args:
            self: Description of self.
            service_session_id: Description of service_session_id.
            session_id: Description of session_id.
        
        Returns:
            Description of the return value.
        """
        return AgentSession(session_id=session_id, service_session_id=service_session_id)

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]]: ...

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Implements run.
        
        Args:
            self: Description of self.
            messages: Description of messages.
            stream: Description of stream.
            session: Description of session.
            function_invocation_kwargs: Description of function_invocation_kwargs.
            client_kwargs: Description of client_kwargs.
        
        Returns:
            Description of the return value.
        """
        if stream:

            async def _stream() -> AsyncIterator[AgentResponseUpdate]:
                """Implements  stream.
                """
                yield AgentResponseUpdate(contents=[Content.from_text(text="ok")], role="assistant")

            return ResponseStream(_stream(), finalizer=lambda updates: AgentResponse.from_updates(updates))

        async def _get_response() -> AgentResponse[Any]:
            """Implements  get response.
            
            Returns:
                Description of the return value.
            """
            return AgentResponse(messages=Message(role="assistant", contents=[Content.from_text(text="ok")]))

        return _get_response()


class TestAgentState:
    """Represents the TestAgentState type and related behavior.
    """
    def test_session_store_is_owned_by_core(self) -> None:
        """Validates behavior for session store is owned by core.
        
        Args:
            self: Description of self.
        """
        assert "SessionStore" not in agent_framework_hosting.__all__
        assert not hasattr(agent_framework_hosting, "SessionStore")

    def test_default_session_store_is_fresh_in_memory_store(self) -> None:
        """Validates behavior for default session store is fresh in memory store.
        
        Args:
            self: Description of self.
        """
        agent = _FakeAgent()
        state = AgentState(agent)

        assert state.target is agent
        assert isinstance(state.session_store, SessionStore)

    def test_accepts_session_store_instance(self) -> None:
        """Validates behavior for accepts session store instance.
        
        Args:
            self: Description of self.
        """
        store = SessionStore()
        state = AgentState(_FakeAgent(), session_store=store)

        assert state.session_store is store

    async def test_callable_target_cached_by_default(self) -> None:
        """Validates behavior for callable target cached by default.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        calls = 0

        def create_agent() -> _FakeAgent:
            """Implements create agent.
            
            Returns:
                Description of the return value.
            """
            nonlocal calls
            calls += 1
            return _FakeAgent()

        state = AgentState(create_agent)

        first = await state.get_target()
        second = await state.get_target()

        assert first is second
        assert calls == 1

    async def test_callable_target_cache_can_be_disabled(self) -> None:
        """Validates behavior for callable target cache can be disabled.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        calls = 0

        def create_agent() -> _FakeAgent:
            """Implements create agent.
            
            Returns:
                Description of the return value.
            """
            nonlocal calls
            calls += 1
            return _FakeAgent()

        state = AgentState(create_agent, cache_target=False)

        first = await state.get_target()
        second = await state.get_target()

        assert first is not second
        assert calls == 2

    async def test_async_callable_target(self) -> None:
        """Validates behavior for async callable target.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        async def create_agent() -> _FakeAgent:
            """Implements create agent.
            
            Returns:
                Description of the return value.
            """
            return _FakeAgent()

        state = AgentState(create_agent)

        assert isinstance(await state.get_target(), _FakeAgent)

    async def test_bare_awaitable_target_is_awaited_once_for_concurrent_callers(self) -> None:
        """Validates behavior for bare awaitable target is awaited once for concurrent callers.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        calls = 0

        async def create_agent() -> _FakeAgent:
            """Implements create agent.
            
            Returns:
                Description of the return value.
            """
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return _FakeAgent()

        state = AgentState(create_agent())

        first, second = await asyncio.gather(state.get_target(), state.get_target())

        assert first is second
        assert calls == 1

    def test_cache_target_false_rejects_bare_awaitable(self) -> None:
        """Validates behavior for cache target false rejects bare awaitable.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        async def create_agent() -> _FakeAgent:
            """Implements create agent.
            
            Returns:
                Description of the return value.
            """
            return _FakeAgent()

        coro = create_agent()
        try:
            with pytest.raises(ValueError, match="cache_target=False"):
                AgentState(coro, cache_target=False)
        finally:
            coro.close()

    async def test_get_or_create_session_creates_and_stores_once(self) -> None:
        """Validates behavior for get or create session creates and stores once.
        
        Args:
            self: Description of self.
        """
        agent = _FakeAgent()
        state = AgentState(agent)

        first = await state.get_or_create_session("session-1")
        second = await state.get_or_create_session("session-1")

        assert first is not second
        assert first.session_id == "session-1"
        assert second.session_id == "session-1"
        assert len(agent.created_sessions) == 1

    @pytest.mark.parametrize("session_id", ["two words", "tenant/user", "tenant:conversation", "' OR 1=1 --"])
    async def test_get_or_create_session_passes_opaque_id_to_store(self, session_id: str) -> None:
        """Validates behavior for get or create session passes opaque id to store.
        
        Args:
            self: Description of self.
            session_id: Description of session_id.
        
        Returns:
            Description of the return value.
        """
        class _RecordingSessionStore(SessionStore):
            """Represents the _RecordingSessionStore type and related behavior.
            """
            def __init__(self) -> None:
                """Initializes a new instance.
                
                Args:
                    self: Description of self.
                """
                super().__init__()
                self.keys: list[str] = []

            async def get(self, session_id: str) -> AgentSession | None:
                """Implements get.
                
                Args:
                    self: Description of self.
                    session_id: Description of session_id.
                
                Returns:
                    Description of the return value.
                """
                self.keys.append(session_id)
                return await super().get(session_id)

            async def set(self, session_id: str, session: AgentSession) -> None:
                """Implements set.
                
                Args:
                    self: Description of self.
                    session_id: Description of session_id.
                    session: Description of session.
                """
                self.keys.append(session_id)
                await super().set(session_id, session)

        agent = _FakeAgent()
        store = _RecordingSessionStore()
        state = AgentState(agent, session_store=store)

        session = await state.get_or_create_session(session_id)

        assert session.session_id == session_id
        assert len(agent.created_sessions) == 1
        assert len(set(store.keys)) == 1
        assert store.keys[0] == session_id

    async def test_get_or_create_session_rejects_empty_id(self) -> None:
        """Validates behavior for get or create session rejects empty id.
        
        Args:
            self: Description of self.
        """
        agent = _FakeAgent()
        state = AgentState(agent)

        with pytest.raises(ValueError, match="non-empty"):
            await state.get_or_create_session("")

        assert agent.created_sessions == []

    async def test_get_or_create_session_creates_once_for_concurrent_callers(self) -> None:
        """Validates behavior for get or create session creates once for concurrent callers.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        class _YieldingSessionStore(SessionStore):
            """Represents the _YieldingSessionStore type and related behavior.
            """
            async def get(self, session_id: str) -> AgentSession | None:
                """Implements get.
                
                Args:
                    self: Description of self.
                    session_id: Description of session_id.
                
                Returns:
                    Description of the return value.
                """
                await asyncio.sleep(0)
                return await super().get(session_id)

            async def set(self, session_id: str, session: AgentSession) -> None:
                """Implements set.
                
                Args:
                    self: Description of self.
                    session_id: Description of session_id.
                    session: Description of session.
                """
                await asyncio.sleep(0)
                await super().set(session_id, session)

        agent = _FakeAgent()
        state = AgentState(agent, session_store=_YieldingSessionStore())

        sessions = await asyncio.gather(*(state.get_or_create_session("session-1") for _ in range(20)))

        assert len({id(session) for session in sessions}) == len(sessions)
        assert all(session.session_id == "session-1" for session in sessions)
        assert len(agent.created_sessions) == 1

    async def test_get_or_create_session_reuses_a_session_set_on_the_state(self) -> None:
        """Validates behavior for get or create session reuses a session set on the state.
        
        Args:
            self: Description of self.
        """
        agent = _FakeAgent()
        state = AgentState(agent)
        pre_existing = AgentSession(session_id="session-1")
        await state.set_session("session-1", pre_existing)

        session = await state.get_or_create_session("session-1")

        assert session is not pre_existing
        assert session.session_id == pre_existing.session_id
        assert len(agent.created_sessions) == 0


class TestWorkflowState:
    """Represents the TestWorkflowState type and related behavior.
    """
    def test_accepts_workflow_target_instance(self) -> None:
        """Validates behavior for accepts workflow target instance.
        
        Args:
            self: Description of self.
        """
        workflow = _workflow_fixture("build_echo_workflow")()
        state: WorkflowState[Workflow] = WorkflowState(workflow)

        assert state.target is workflow

    async def test_workflow_target_resolved_from_factory(self) -> None:
        """Validates behavior for workflow target resolved from factory.
        
        Args:
            self: Description of self.
        """
        build_echo_workflow = _workflow_fixture("build_echo_workflow")

        state: WorkflowState[Workflow] = WorkflowState(build_echo_workflow)

        target = await state.get_target()
        assert isinstance(target, Workflow)

    async def test_bare_awaitable_workflow_target_is_awaited_once_for_concurrent_callers(self) -> None:
        """Validates behavior for bare awaitable workflow target is awaited once for concurrent callers.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        calls = 0

        async def create_workflow() -> Workflow:
            """Implements create workflow.
            
            Returns:
                Description of the return value.
            """
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return _workflow_fixture("build_echo_workflow")()

        state: WorkflowState[Workflow] = WorkflowState(create_workflow())

        first, second = await asyncio.gather(state.get_target(), state.get_target())

        assert first is second
        assert calls == 1

    async def test_accepts_workflow_builder_instance_directly(self) -> None:
        """A ``WorkflowBuilder`` is not itself callable or awaitable; the state must
        recognize its `build()` method and call it, not cache the raw builder."""
        builder = _workflow_fixture("echo_workflow_builder")()

        state: WorkflowState[Workflow] = WorkflowState(builder)

        target = await state.get_target()
        assert isinstance(target, Workflow)
        assert state.target is target

    async def test_workflow_builder_is_built_once_and_cached_by_default(self) -> None:
        """Validates behavior for workflow builder is built once and cached by default.
        
        Args:
            self: Description of self.
        """
        builder = _workflow_fixture("echo_workflow_builder")()
        state: WorkflowState[Workflow] = WorkflowState(builder)

        first = await state.get_target()
        second = await state.get_target()

        assert first is second

    async def test_workflow_builder_returns_fresh_targets_when_cache_disabled(self) -> None:
        """Validates behavior for workflow builder returns fresh targets when cache disabled.
        
        Args:
            self: Description of self.
        """
        builder = _workflow_fixture("echo_workflow_builder")()
        state: WorkflowState[Workflow] = WorkflowState(builder, cache_target=False)

        first = await state.get_target()
        second = await state.get_target()

        assert first is not second

    async def test_accepts_orchestration_style_builder_without_importing_orchestrations(self) -> None:
        """``SupportsBuild`` is structural: any object with a zero-arg ``build() -> Workflow``
        is accepted, matching ``agent_framework_orchestrations``' builders without this
        package depending on that one."""
        workflow = _workflow_fixture("build_echo_workflow")()

        class _FakeOrchestrationBuilder:
            """Represents the _FakeOrchestrationBuilder type and related behavior.
            """
            def build(self) -> Workflow:
                """Implements build.
                
                Args:
                    self: Description of self.
                
                Returns:
                    Description of the return value.
                """
                return workflow

        state: WorkflowState[Workflow] = WorkflowState(_FakeOrchestrationBuilder())

        assert await state.get_target() is workflow
