# Copyright (c) Microsoft. All rights reserved.

"""Tests for the functional workflow API (@workflow, @step, RunContext)."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, overload

import pytest

from agent_framework import (
    AgentResponseUpdate,
    CheckpointStorage,
    ExperimentalFeature,
    FunctionalWorkflow,
    FunctionalWorkflowAgent,
    FunctionalWorkflowDefinition,
    InMemoryCheckpointStorage,
    RunContext,
    StepWrapper,
    WorkflowEvent,
    WorkflowEventSource,
    WorkflowRunResult,
    WorkflowRunState,
    get_run_context,
    step,
    workflow,
)
from agent_framework._workflows._functional import (
    RunContext as _RunContext,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@overload
def built_workflow(func: Callable[..., Awaitable[Any]]) -> FunctionalWorkflow: ...


@overload
def built_workflow(
    *,
    name: str | None = None,
    description: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], FunctionalWorkflow]: ...


def built_workflow(
    func: Callable[..., Awaitable[Any]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
) -> FunctionalWorkflow | Callable[[Callable[..., Awaitable[Any]]], FunctionalWorkflow]:
    """Build a fresh executable workflow for behavior-focused tests."""

    def decorate(fn: Callable[..., Awaitable[Any]]) -> FunctionalWorkflow:
        """Implements decorate.
        
        Args:
            fn: Description of fn.
        
        Returns:
            Description of the return value.
        """
        return workflow(name=name, description=description)(fn).build(checkpoint_storage=checkpoint_storage)

    return decorate(func) if func is not None else decorate


@step
async def add_one(x: int) -> int:
    """Implements add one.
    
    Args:
        x: Description of x.
    
    Returns:
        Description of the return value.
    """
    return x + 1


@step
async def double(x: int) -> int:
    """Implements double.
    
    Args:
        x: Description of x.
    
    Returns:
        Description of the return value.
    """
    return x * 2


@step
async def to_upper(s: str) -> str:
    """Implements to upper.
    
    Args:
        s: Description of s.
    
    Returns:
        Description of the return value.
    """
    return s.upper()


@step(name="custom_name")
async def named_step(x: int) -> int:
    """Implements named step.
    
    Args:
        x: Description of x.
    
    Returns:
        Description of the return value.
    """
    return x + 10


@step
async def failing_step(x: int) -> int:
    """Implements failing step.
    
    Args:
        x: Description of x.
    """
    raise ValueError(f"step failed with {x}")


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    """Represents the TestBasicExecution type and related behavior.
    """
    async def test_simple_sequential_pipeline(self):
        """Validates behavior for simple sequential pipeline.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a = await add_one(x)
            return await double(a)

        result = await pipeline.run(5)
        assert isinstance(result, WorkflowRunResult)
        outputs = result.get_outputs()
        assert outputs == [12]  # (5+1)*2

    async def test_workflow_with_string_data(self):
        """Validates behavior for workflow with string data.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def upper_pipeline(text: str) -> str:
            """Implements upper pipeline.
            
            Args:
                text: Description of text.
            
            Returns:
                Description of the return value.
            """
            return await to_upper(text)

        result = await upper_pipeline.run("hello")
        assert result.get_outputs() == ["HELLO"]

    async def test_workflow_returns_result(self):
        """Validates behavior for workflow returns result.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def simple(x: int) -> int:
            """Implements simple.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        result = await simple.run(10)
        assert result.get_outputs() == [11]

    async def test_workflow_name_defaults_to_function_name(self):
        """Validates behavior for workflow name defaults to function name.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def my_pipeline(x: int) -> int:
            """Implements my pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        assert my_pipeline.name == "my_pipeline"

    async def test_workflow_custom_name(self):
        """Validates behavior for workflow custom name.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow(name="custom_wf", description="A test workflow")
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        assert wf.name == "custom_wf"
        assert wf.description == "A test workflow"


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class TestEventEmission:
    """Represents the TestEventEmission type and related behavior.
    """
    async def test_step_events_emitted(self):
        """Validates behavior for step events emitted.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        result = await pipeline.run(5)
        event_types = [e.type for e in result]
        assert "executor_invoked" in event_types
        assert "executor_completed" in event_types
        assert "output" in event_types

    async def test_step_events_carry_executor_id(self):
        """Validates behavior for step events carry executor id.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        result = await pipeline.run(5)
        invoked_events = [e for e in result if e.type == "executor_invoked"]
        assert len(invoked_events) == 1
        assert invoked_events[0].executor_id == "add_one"

        completed_events = [e for e in result if e.type == "executor_completed"]
        assert len(completed_events) == 1
        assert completed_events[0].executor_id == "add_one"
        assert completed_events[0].data == 6

    async def test_status_events_in_timeline(self):
        """Validates behavior for status events in timeline.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        result = await pipeline.run(1)
        states = [e.state for e in result.status_timeline()]
        assert WorkflowRunState.IN_PROGRESS in states
        assert WorkflowRunState.IDLE in states

    async def test_final_state_is_idle(self):
        """Validates behavior for final state is idle.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        result = await pipeline.run(1)
        assert result.get_final_state() == WorkflowRunState.IDLE

    async def test_custom_event(self):
        """Validates behavior for custom event.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        from agent_framework import WorkflowEvent

        @built_workflow
        async def pipeline(x: int, ctx: RunContext) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            await ctx.add_event(WorkflowEvent("intermediate", executor_id="pipeline", data="custom_data"))
            return x

        result = await pipeline.run(1)
        intermediate_events = [e for e in result if e.type == "intermediate"]
        assert len(intermediate_events) == 1
        assert intermediate_events[0].data == "custom_data"


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Represents the TestParallelExecution type and related behavior.
    """
    async def test_parallel_tasks_with_gather(self):
        """Validates behavior for parallel tasks with gather.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @step
        async def slow_add(x: int) -> int:
            """Implements slow add.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            await asyncio.sleep(0.01)
            return x + 1

        @step
        async def slow_double(x: int) -> int:
            """Implements slow double.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            await asyncio.sleep(0.01)
            return x * 2

        @built_workflow
        async def parallel_wf(x: int) -> list[int]:
            """Implements parallel wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a, b = await asyncio.gather(slow_add(x), slow_double(x))
            return [a, b]

        result = await parallel_wf.run(5)
        outputs = result.get_outputs()
        assert outputs == [[6, 10]]

    async def test_parallel_events_all_emitted(self):
        """Validates behavior for parallel events all emitted.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @step
        async def task_a(x: int) -> int:
            """Implements task a.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 1

        @step
        async def task_b(x: int) -> int:
            """Implements task b.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 2

        @built_workflow
        async def par_wf(x: int) -> tuple[int, int]:
            """Implements par wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a, b = await asyncio.gather(task_a(x), task_b(x))
            return (a, b)

        result = await par_wf.run(3)
        invoked = [e for e in result if e.type == "executor_invoked"]
        completed = [e for e in result if e.type == "executor_completed"]
        assert len(invoked) == 2
        assert len(completed) == 2


# ---------------------------------------------------------------------------
# HITL (request_info / resume)
# ---------------------------------------------------------------------------


class TestHITL:
    """Represents the TestHITL type and related behavior.
    """
    async def test_workflow_definition_builds_isolated_pending_continuations(self):
        """Validates behavior for workflow definition builds isolated pending continuations.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @workflow
        async def review_wf(doc: str, ctx: RunContext) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info(doc, response_type=str)
            return f"{doc}:{feedback}"

        assert not hasattr(review_wf, "run")

        caller_a = review_wf.build()
        caller_b = review_wf.build()

        caller_a_paused = await caller_a.run("caller-a")
        request_id = caller_a_paused.get_request_info_events()[0].request_id

        with pytest.raises(ValueError, match="no pending request_info events"):
            await caller_b.run(responses={request_id: "caller-b-response"})

        caller_a_completed = await caller_a.run(responses={request_id: "caller-a-response"})
        assert caller_a_completed.get_outputs() == ["caller-a:caller-a-response"]

    async def test_build_does_not_inherit_checkpoint_storage_by_default(self):
        """Validates behavior for build does not inherit checkpoint storage by default.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @workflow
        async def review_wf(doc: str) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
            
            Returns:
                Description of the return value.
            """
            return doc

        caller = review_wf.build()

        with pytest.raises(ValueError, match="checkpoint_storage"):
            await caller.run(checkpoint_id="missing")

    async def test_build_accepts_tenant_scoped_checkpoint_storage(self):
        """Validates behavior for build accepts tenant scoped checkpoint storage.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        caller_storage = InMemoryCheckpointStorage()

        @workflow
        async def review_wf(doc: str, ctx: RunContext) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return await ctx.request_info(doc, response_type=str)

        caller = review_wf.build(checkpoint_storage=caller_storage)
        await caller.run("caller")

        assert len(await caller_storage.list_checkpoints(workflow_name="review_wf")) == 1

    async def test_request_info_interrupts(self):
        """Validates behavior for request info interrupts.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def review_wf(doc: str, ctx: RunContext) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"draft": doc}, response_type=str, request_id="req1")
            return f"Final: {feedback}"

        # Phase 1: should interrupt with pending request
        result = await review_wf.run("my doc")
        assert result.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        request_events = result.get_request_info_events()
        assert len(request_events) == 1
        assert request_events[0].request_id == "req1"

    async def test_request_info_resume(self):
        """Validates behavior for request info resume.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def review_wf(doc: str, ctx: RunContext) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"draft": doc}, response_type=str, request_id="req1")
            return f"Final: {feedback}"

        # Phase 1
        result1 = await review_wf.run("my doc")
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # Phase 2: resume with response
        result2 = await review_wf.run(responses={"req1": "Looks great!"})
        outputs = result2.get_outputs()
        assert outputs == ["Final: Looks great!"]
        assert result2.get_final_state() == WorkflowRunState.IDLE

    async def test_fresh_message_while_pending_requests_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A fresh message while request_info events are pending is allowed but logs a warning."""

        @built_workflow
        async def review_wf(doc: str, ctx: RunContext) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"draft": doc}, response_type=str, request_id="req1")
            return f"Final: {feedback}"

        result1 = await review_wf.run("my doc")
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # Starting fresh input while a request is pending does not abandon it, but advances
        # workflow state so a later response may apply to a moved-on workflow -> warn (but proceed).
        with caplog.at_level(logging.WARNING):
            await review_wf.run("another doc")

        assert "request_info event(s) are still pending" in caplog.text
        assert "a fresh message" in caplog.text

    async def test_responses_while_pending_requests_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Delivering responses is the normal completion path and must not warn."""

        @built_workflow
        async def review_wf(doc: str, ctx: RunContext) -> str:
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"draft": doc}, response_type=str, request_id="req1")
            return f"Final: {feedback}"

        result1 = await review_wf.run("my doc")
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result2 = await review_wf.run(responses={"req1": "Looks great!"})

        assert result2.get_final_state() == WorkflowRunState.IDLE
        assert "still pending" not in caplog.text

    async def test_untyped_ctx_parameter(self):
        """ctx is injected by parameter name even without a RunContext annotation."""

        @built_workflow  # pyright: ignore[reportUnknownArgumentType]
        async def review_wf(doc: str, ctx) -> str:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
            """Implements review wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback: str = await ctx.request_info({"draft": doc}, response_type=str, request_id="req1")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return f"Final: {feedback}"

        result1 = await review_wf.run("my doc")
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        result2 = await review_wf.run(responses={"req1": "LGTM"})
        assert result2.get_outputs() == ["Final: LGTM"]

    async def test_multiple_sequential_interrupts(self):
        """Validates behavior for multiple sequential interrupts.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def multi_hitl(data: str, ctx: RunContext) -> str:
            """Implements multi hitl.
            
            Args:
                data: Description of data.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            r1 = await ctx.request_info("step1", response_type=str, request_id="r1")
            r2 = await ctx.request_info("step2", response_type=str, request_id="r2")
            return f"{r1}+{r2}"

        # Phase 1: first interrupt
        result1 = await multi_hitl.run("start")
        assert len(result1.get_request_info_events()) == 1
        assert result1.get_request_info_events()[0].request_id == "r1"

        # Phase 2: respond to first, hits second
        result2 = await multi_hitl.run(responses={"r1": "A"})
        assert len(result2.get_request_info_events()) == 1
        assert result2.get_request_info_events()[0].request_id == "r2"

        # Phase 3: respond to second
        result3 = await multi_hitl.run(responses={"r1": "A", "r2": "B"})
        assert result3.get_outputs() == ["A+B"]

    async def test_request_info_auto_generates_id(self):
        """Validates behavior for request info auto generates id.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def auto_id_wf(x: int, ctx: RunContext) -> None:
            """Implements auto id wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            """
            await ctx.request_info("need data", response_type=str)

        result = await auto_id_wf.run(1)
        events = result.get_request_info_events()
        assert len(events) == 1
        assert events[0].request_id  # should be a non-empty uuid string


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Represents the TestErrorHandling type and related behavior.
    """
    async def test_step_failure_propagates(self):
        """Validates behavior for step failure propagates.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def failing_wf(x: int) -> None:
            """Implements failing wf.
            
            Args:
                x: Description of x.
            """
            await failing_step(x)

        with pytest.raises(ValueError, match="step failed with 42"):
            await failing_wf.run(42)

    async def test_step_failure_emits_executor_failed(self):
        """Validates behavior for step failure emits executor failed.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def failing_wf(x: int) -> None:
            """Implements failing wf.
            
            Args:
                x: Description of x.
            """
            await failing_step(x)

        # Use stream to collect events before the raise
        stream = failing_wf.run(42, stream=True)
        events: list[WorkflowEvent[object]] = []
        with pytest.raises(ValueError):
            async for event in stream:
                events.append(event)

        failed_events = [e for e in events if e.type == "executor_failed"]
        assert len(failed_events) == 1
        assert failed_events[0].executor_id == "failing_step"

    async def test_workflow_failure_emits_failed_status(self):
        """Validates behavior for workflow failure emits failed status.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def bad_wf(x: int) -> None:
            """Implements bad wf.
            
            Args:
                x: Description of x.
            """
            raise RuntimeError("workflow broke")

        stream = bad_wf.run(42, stream=True)
        events: list[WorkflowEvent[object]] = []
        with pytest.raises(RuntimeError, match="workflow broke"):
            async for event in stream:
                events.append(event)

        failed_events = [e for e in events if e.type == "failed"]
        assert len(failed_events) == 1
        status_events = [e for e in events if e.type == "status"]
        assert any(e.state == WorkflowRunState.FAILED for e in status_events)

    async def test_invalid_params_message_and_responses(self):
        """Validates behavior for invalid params message and responses.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def wf(x: int) -> None:
            """Implements wf.
            
            Args:
                x: Description of x.
            """
            pass

        with pytest.raises(ValueError, match="Cannot provide both"):
            await wf.run("hello", responses={"r1": "val"})

    async def test_invalid_params_message_and_checkpoint(self):
        """Validates behavior for invalid params message and checkpoint.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def wf(x: int) -> None:
            """Implements wf.
            
            Args:
                x: Description of x.
            """
            pass

        with pytest.raises(ValueError, match="Cannot provide both"):
            await wf.run("hello", checkpoint_id="abc")

    async def test_invalid_params_nothing(self):
        """Validates behavior for invalid params nothing.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def wf(x: int) -> None:
            """Implements wf.
            
            Args:
                x: Description of x.
            """
            pass

        with pytest.raises(ValueError, match="Must provide at least one"):
            await wf.run()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """Represents the TestStreaming type and related behavior.
    """
    async def test_streaming_yields_events(self):
        """Validates behavior for streaming yields events.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        stream = pipeline.run(5, stream=True)
        events: list[WorkflowEvent[object]] = []
        async for event in stream:
            events.append(event)

        event_types = [e.type for e in events]
        assert "started" in event_types
        assert "executor_invoked" in event_types
        assert "executor_completed" in event_types
        assert "output" in event_types

    async def test_streaming_final_response(self):
        """Validates behavior for streaming final response.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        stream = pipeline.run(5, stream=True)
        final = await stream.get_final_response()
        assert isinstance(final, WorkflowRunResult)
        assert final.get_outputs() == [6]

    async def test_streaming_context_reports_streaming(self):
        """Validates behavior for streaming context reports streaming.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        streaming_flag = None

        @built_workflow
        async def wf(x: int, ctx: RunContext) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            nonlocal streaming_flag
            streaming_flag = ctx.is_streaming()  # type: ignore[assignment]
            return x

        stream = wf.run(1, stream=True)
        await stream.get_final_response()
        assert streaming_flag is True

        streaming_flag = None
        await wf.run(1)
        assert streaming_flag is False

    async def test_abandoned_stream_finalizes_without_event_loop_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Breaking out of a streaming run must not leak ContextVar tokens on GC.

        Regression for https://github.com/microsoft/agent-framework/issues/7787:
        the run span and ``_framework_event_origin()`` used to stay open across
        event yields, so abandoning the stream and letting it be garbage-collected
        reset those tokens from a different Context.
        """
        loop = asyncio.get_running_loop()
        loop_errors: list[BaseException] = []
        original_handler = loop.get_exception_handler()

        def _capture_loop_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            """Implements  capture loop exception.
            
            Args:
                _loop: Description of _loop.
                context: Description of context.
            """
            exc = context.get("exception")
            if isinstance(exc, BaseException):
                loop_errors.append(exc)
            if original_handler is not None:
                original_handler(_loop, context)

        loop.set_exception_handler(_capture_loop_exception)

        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        try:
            with caplog.at_level(logging.ERROR, logger="opentelemetry"):
                stream = pipeline.run(5, stream=True)
                async for _event in stream:
                    break

                del stream
                gc.collect()
                for _ in range(5):
                    await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(original_handler)

        assert loop_errors == [], f"Abandoned stream leaked loop exceptions: {loop_errors!r}"
        otel_errors = [
            rec.getMessage()
            for rec in caplog.records
            if "Failed to detach context" in rec.getMessage()
            or "was created in a different Context" in rec.getMessage()
        ]
        assert otel_errors == [], f"Abandoned stream leaked OpenTelemetry errors: {otel_errors!r}"

        follow_up = await pipeline.run(6)
        assert follow_up.get_outputs() == [7]

    async def test_nested_processing_span_parents_under_workflow_run(self, span_exporter: Any) -> None:
        """Spans opened during ``_execute`` must parent under the unattached ``workflow.run`` span."""
        from agent_framework.observability import OtelAttr, create_processing_span

        @step
        async def traced_add(x: int) -> int:
            """Implements traced add.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            with create_processing_span("traced_add", "StepWrapper", "int", "int"):
                return x + 1

        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await traced_add(x)

        span_exporter.clear()  # type: ignore[attr-defined]
        result = await pipeline.run(5)
        assert result.get_outputs() == [6]

        spans = span_exporter.get_finished_spans()  # type: ignore[attr-defined]
        run_spans = [s for s in spans if s.name == OtelAttr.WORKFLOW_RUN_SPAN]
        process_spans = [s for s in spans if s.name == f"{OtelAttr.EXECUTOR_PROCESS_SPAN} traced_add"]
        assert len(run_spans) == 1
        assert len(process_spans) == 1
        process_parent = process_spans[0].parent
        assert process_parent is not None
        assert process_parent.span_id == run_spans[0].context.span_id

    async def test_started_event_origin_is_framework_after_origin_manager_closes(self) -> None:
        """Framework lifecycle events stay tagged FRAMEWORK even when yielded outside the origin CM."""

        @built_workflow
        async def pipeline(x: int) -> int:
            """Implements pipeline.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        stream = pipeline.run(5, stream=True)
        started = await anext(aiter(stream))
        assert started.type == "started"
        assert started.origin == WorkflowEventSource.FRAMEWORK
        await stream.get_final_response()


# ---------------------------------------------------------------------------
# Step passthrough outside workflow
# ---------------------------------------------------------------------------


class TestStepPassthrough:
    """Represents the TestStepPassthrough type and related behavior.
    """
    async def test_step_works_outside_workflow(self):
        """Validates behavior for step works outside workflow.
        
        Args:
            self: Description of self.
        """
        result = await add_one(10)
        assert result == 11

    async def test_named_step_outside_workflow(self):
        """Validates behavior for named step outside workflow.
        
        Args:
            self: Description of self.
        """
        result = await named_step(5)
        assert result == 15

    def test_step_wrapper_name(self):
        """Validates behavior for step wrapper name.
        
        Args:
            self: Description of self.
        """
        assert add_one.name == "add_one"
        assert named_step.name == "custom_name"

    def test_step_wrapper_is_step_wrapper(self):
        """Validates behavior for step wrapper is step wrapper.
        
        Args:
            self: Description of self.
        """
        assert isinstance(add_one, StepWrapper)
        assert isinstance(named_step, StepWrapper)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    """Represents the TestStateManagement type and related behavior.
    """
    async def test_get_set_state(self):
        """Validates behavior for get set state.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def stateful_wf(x: int, ctx: RunContext) -> int:
            """Implements stateful wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            ctx.set_state("counter", x)
            return ctx.get_state("counter")

        result = await stateful_wf.run(42)
        assert result.get_outputs() == [42]

    async def test_get_state_default(self):
        """Validates behavior for get state default.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return ctx.get_state("missing", "default_val")

        result = await wf.run(1)
        assert result.get_outputs() == ["default_val"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Represents the TestCheckpointing type and related behavior.
    """
    async def test_checkpoint_save_and_restore(self):
        """Validates behavior for checkpoint save and restore.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()

        @step
        async def expensive(x: int) -> int:
            """Implements expensive.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 100

        @built_workflow(checkpoint_storage=storage)
        async def ckpt_wf(x: int) -> int:
            """Implements ckpt wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await expensive(x)

        result = await ckpt_wf.run(5)
        assert result.get_outputs() == [500]

        # Verify checkpoints were saved: 1 per-step + 1 final
        checkpoints = await storage.list_checkpoints(workflow_name="ckpt_wf")
        assert len(checkpoints) == 2

    async def test_checkpoint_runtime_storage_override(self):
        """Validates behavior for checkpoint runtime storage override.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()

        @step
        async def compute(x: int) -> int:
            """Implements compute.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 1

        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await compute(x)

        result = await wf.run(10, checkpoint_storage=storage)
        assert result.get_outputs() == [11]
        # 1 per-step checkpoint + 1 final checkpoint
        checkpoints = await storage.list_checkpoints(workflow_name="wf")
        assert len(checkpoints) == 2

    async def test_checkpoint_restore_replays_cached_tasks(self):
        """Validates behavior for checkpoint restore replays cached tasks.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()
        call_count = 0

        @step
        async def counting_task(x: int) -> int:
            """Implements counting task.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            nonlocal call_count
            call_count += 1
            return x + 1

        @built_workflow(checkpoint_storage=storage)
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await counting_task(x)

        # First run
        result1 = await wf.run(5)
        assert result1.get_outputs() == [6]
        assert call_count == 1

        # Get checkpoint ID
        checkpoints = await storage.list_checkpoints(workflow_name="wf")
        ckpt_id = checkpoints[0].checkpoint_id

        # Restore — step should replay from cache
        result2 = await wf.run(checkpoint_id=ckpt_id)
        assert result2.get_outputs() == [6]
        assert call_count == 1  # not called again

    async def test_checkpoint_hitl_resume(self):
        """Validates behavior for checkpoint hitl resume.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()

        @built_workflow(checkpoint_storage=storage)
        async def hitl_wf(doc: str, ctx: RunContext) -> str:
            """Implements hitl wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"draft": doc}, response_type=str, request_id="req1")
            return f"Done: {feedback}"

        # Phase 1: interrupt
        result1 = await hitl_wf.run("draft text")
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # Get checkpoint
        checkpoints = await storage.list_checkpoints(workflow_name="hitl_wf")
        ckpt_id = checkpoints[0].checkpoint_id

        # Phase 2: restore and respond
        result2 = await hitl_wf.run(checkpoint_id=ckpt_id, responses={"req1": "Approved!"})
        assert result2.get_outputs() == ["Done: Approved!"]

    async def test_checkpoint_without_storage_raises(self):
        """Validates behavior for checkpoint without storage raises.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        with pytest.raises(ValueError, match="checkpoint_storage"):
            await wf.run(checkpoint_id="nonexistent")

    async def test_checkpoint_preserves_state(self):
        """Validates behavior for checkpoint preserves state.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()

        @built_workflow(checkpoint_storage=storage)
        async def stateful_wf(x: int, ctx: RunContext) -> str:
            """Implements stateful wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            ctx.set_state("key", "value")
            feedback = await ctx.request_info("need info", response_type=str, request_id="r1")
            val = ctx.get_state("key")
            return f"{val}:{feedback}"

        # Phase 1
        result1 = await stateful_wf.run(1)
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # Phase 2: restore and respond
        checkpoints = await storage.list_checkpoints(workflow_name="stateful_wf")
        ckpt_id = checkpoints[0].checkpoint_id

        result2 = await stateful_wf.run(checkpoint_id=ckpt_id, responses={"r1": "hello"})
        assert result2.get_outputs() == ["value:hello"]

    async def test_per_step_checkpoint_enables_crash_recovery(self):
        """Simulates crash recovery: step 1 completes and is checkpointed,
        then the workflow crashes in step 2. Restoring from the per-step
        checkpoint should replay step 1 from cache without re-executing it."""
        storage = InMemoryCheckpointStorage()
        step1_calls = 0
        step2_calls = 0

        @step
        async def slow_step1(x: int) -> int:
            """Implements slow step1.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            nonlocal step1_calls
            step1_calls += 1
            return x + 10

        @step
        async def crashing_step2(x: int) -> int:
            """Implements crashing step2.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            nonlocal step2_calls
            step2_calls += 1
            if step2_calls == 1:
                raise RuntimeError("simulated crash")
            return x * 2

        @built_workflow(checkpoint_storage=storage)
        async def crash_wf(x: int) -> int:
            """Implements crash wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a = await slow_step1(x)
            return await crashing_step2(a)

        # First run: step1 succeeds and checkpoints, step2 crashes
        with pytest.raises(RuntimeError, match="simulated crash"):
            await crash_wf.run(5)

        assert step1_calls == 1
        assert step2_calls == 1

        # A per-step checkpoint was saved after step1 completed
        checkpoints = await storage.list_checkpoints(workflow_name="crash_wf")
        assert len(checkpoints) >= 1
        ckpt_id = checkpoints[0].checkpoint_id

        # Restore from checkpoint: step1 replays from cache, step2 runs fresh
        result = await crash_wf.run(checkpoint_id=ckpt_id)
        assert result.get_outputs() == [30]  # (5+10)*2
        assert step1_calls == 1  # NOT called again — replayed from cache
        assert step2_calls == 2  # called again, succeeds this time

    async def test_per_step_checkpoint_chain(self):
        """Each step creates a new checkpoint chained to the previous one."""
        storage = InMemoryCheckpointStorage()

        @step
        async def s1(x: int) -> int:
            """Implements s1.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 1

        @step
        async def s2(x: int) -> int:
            """Implements s2.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 2

        @step
        async def s3(x: int) -> int:
            """Implements s3.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 3

        @built_workflow(checkpoint_storage=storage)
        async def multi_step_wf(x: int) -> int:
            """Implements multi step wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a = await s1(x)
            b = await s2(a)
            return await s3(b)

        result = await multi_step_wf.run(0)
        assert result.get_outputs() == [6]  # 0+1+2+3

        # 3 per-step checkpoints + 1 final = 4
        checkpoints = await storage.list_checkpoints(workflow_name="multi_step_wf")
        assert len(checkpoints) == 4

    async def test_no_checkpoint_on_cache_hit(self):
        """During replay, cached steps should NOT create additional checkpoints."""
        storage = InMemoryCheckpointStorage()

        @step
        async def compute(x: int) -> int:
            """Implements compute.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 1

        @built_workflow(checkpoint_storage=storage)
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await compute(x)

        # First run: 1 per-step + 1 final = 2 checkpoints
        await wf.run(5)
        checkpoints = await storage.list_checkpoints(workflow_name="wf")
        assert len(checkpoints) == 2
        ckpt_id = checkpoints[0].checkpoint_id

        # Restore: step replays from cache (no new per-step checkpoint),
        # but final checkpoint still saved = 1 new checkpoint
        await wf.run(checkpoint_id=ckpt_id)
        checkpoints = await storage.list_checkpoints(workflow_name="wf")
        assert len(checkpoints) == 3  # 2 from first run + 1 final from restore


# ---------------------------------------------------------------------------
# Branching / control flow
# ---------------------------------------------------------------------------


class TestControlFlow:
    """Represents the TestControlFlow type and related behavior.
    """
    async def test_if_else_branching(self):
        """Validates behavior for if else branching.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @dataclass
        class Classification:
            """Represents the Classification type and related behavior.
            """
            is_spam: bool

        @step
        async def classify(text: str) -> Classification:
            """Implements classify.
            
            Args:
                text: Description of text.
            
            Returns:
                Description of the return value.
            """
            return Classification(is_spam="spam" in text.lower())

        @step
        async def process_normal(text: str) -> str:
            """Implements process normal.
            
            Args:
                text: Description of text.
            
            Returns:
                Description of the return value.
            """
            return f"processed: {text}"

        @step
        async def quarantine(text: str) -> str:
            """Implements quarantine.
            
            Args:
                text: Description of text.
            
            Returns:
                Description of the return value.
            """
            return f"quarantined: {text}"

        @built_workflow
        async def email_pipeline(email: str) -> str:
            """Implements email pipeline.
            
            Args:
                email: Description of email.
            
            Returns:
                Description of the return value.
            """
            cl = await classify(email)
            if cl.is_spam:
                result = await quarantine(email)
            else:
                result = await process_normal(email)
            return result

        result_spam = await email_pipeline.run("Buy spam now!")
        assert result_spam.get_outputs() == ["quarantined: Buy spam now!"]

        result_normal = await email_pipeline.run("Hello friend")
        assert result_normal.get_outputs() == ["processed: Hello friend"]


# ---------------------------------------------------------------------------
# Nested workflow calls
# ---------------------------------------------------------------------------


class TestNestedWorkflows:
    """Represents the TestNestedWorkflows type and related behavior.
    """
    async def test_nested_workflow_as_task(self):
        """Validates behavior for nested workflow as task.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @step
        async def step_a(x: int) -> int:
            """Implements step a.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 1

        @built_workflow
        async def inner_wf(x: int) -> int:
            """Implements inner wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await step_a(x)

        @step
        async def call_inner(x: int) -> int:
            """Implements call inner.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            result = await inner_wf.run(x)
            return result.get_outputs()[0]

        @built_workflow
        async def outer_wf(x: int) -> int:
            """Implements outer wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await call_inner(x)

        result = await outer_wf.run(5)
        assert result.get_outputs() == [6]


# ---------------------------------------------------------------------------
# as_agent()
# ---------------------------------------------------------------------------


class TestAsAgent:
    """Represents the TestAsAgent type and related behavior.
    """
    async def test_as_agent_returns_agent(self):
        """Validates behavior for as agent returns agent.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return f"result: {x}"

        agent = wf.as_agent()
        assert agent.name == "wf"

    async def test_as_agent_custom_name(self):
        """Validates behavior for as agent custom name.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        agent = wf.as_agent(name="my_agent")
        assert agent.name == "my_agent"

    async def test_as_agent_run(self):
        """Validates behavior for as agent run.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await add_one(x)

        agent = wf.as_agent()
        response = await agent.run(10)
        assert response.text == "11"

    async def test_as_agent_run_streaming(self):
        """Validates behavior for as agent run streaming.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return f"result: {x}"

        agent = wf.as_agent()
        stream = agent.run(10, stream=True)
        updates: list[AgentResponseUpdate] = []
        async for update in stream:
            updates.append(update)
        assert len(updates) == 1
        assert updates[0].text == "result: 10"

        response = await stream.get_final_response()
        assert len(response.messages) >= 1

    async def test_as_agent_has_id_and_description(self):
        """Validates behavior for as agent has id and description.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow(description="A test workflow")
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        agent = wf.as_agent(name="my_agent")
        assert agent.id == "FunctionalWorkflowAgent_my_agent"
        assert agent.description == "A test workflow"


# ---------------------------------------------------------------------------
# Concurrent execution guard
# ---------------------------------------------------------------------------


class TestConcurrencyGuard:
    """Represents the TestConcurrencyGuard type and related behavior.
    """
    async def test_concurrent_run_raises(self):
        """Validates behavior for concurrent run raises.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def slow_wf(x: int) -> int:
            """Implements slow wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            await asyncio.sleep(0.1)
            return x

        # Start first run
        stream = slow_wf.run(1, stream=True)

        # Try to start second run while first is active
        with pytest.raises(RuntimeError, match="already running"):
            slow_wf.run(2, stream=True)

        # Consume the stream to clean up
        await stream.get_final_response()

    async def test_run_after_completion(self):
        """Validates behavior for run after completion.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        result1 = await wf.run(1)
        assert result1.get_outputs() == [1]

        # Should be able to run again after first completes
        result2 = await wf.run(2)
        assert result2.get_outputs() == [2]


# ---------------------------------------------------------------------------
# Decorator forms
# ---------------------------------------------------------------------------


class TestDecoratorForms:
    """Represents the TestDecoratorForms type and related behavior.
    """
    def test_step_bare_decorator(self):
        """Validates behavior for step bare decorator.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @step
        async def my_step(x: int) -> int:
            """Implements my step.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        assert isinstance(my_step, StepWrapper)
        assert my_step.name == "my_step"

    def test_step_with_name(self):
        """Validates behavior for step with name.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @step(name="renamed")
        async def my_step(x: int) -> int:
            """Implements my step.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        assert isinstance(my_step, StepWrapper)
        assert my_step.name == "renamed"

    def test_workflow_bare_decorator(self):
        """Validates behavior for workflow bare decorator.
        
        Args:
            self: Description of self.
        """
        @workflow
        async def my_wf(x: int) -> None:
            """Implements my wf.
            
            Args:
                x: Description of x.
            """
            pass

        assert isinstance(my_wf, FunctionalWorkflowDefinition)
        assert my_wf.name == "my_wf"

    def test_workflow_with_params(self):
        """Validates behavior for workflow with params.
        
        Args:
            self: Description of self.
        """
        @workflow(name="custom", description="desc")
        async def my_wf(x: int) -> None:
            """Implements my wf.
            
            Args:
                x: Description of x.
            """
            pass

        assert isinstance(my_wf, FunctionalWorkflowDefinition)
        assert my_wf.name == "custom"
        assert my_wf.description == "desc"


# ---------------------------------------------------------------------------
# include_status_events
# ---------------------------------------------------------------------------


class TestIncludeStatusEvents:
    """Represents the TestIncludeStatusEvents type and related behavior.
    """
    async def test_status_events_excluded_by_default(self):
        """Validates behavior for status events excluded by default.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        result = await wf.run(1)
        status_in_list = [e for e in result if e.type == "status"]
        assert len(status_in_list) == 0

    async def test_status_events_included_when_requested(self):
        """Validates behavior for status events included when requested.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        result = await wf.run(1, include_status_events=True)
        status_in_list = [e for e in result if e.type == "status"]
        assert len(status_in_list) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Represents the TestEdgeCases type and related behavior.
    """
    async def test_workflow_with_no_tasks(self):
        """Validates behavior for workflow with no tasks.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def no_tasks(x: int) -> int:
            """Implements no tasks.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 2

        result = await no_tasks.run(5)
        assert result.get_outputs() == [10]

    async def test_workflow_with_no_output(self):
        """Validates behavior for workflow with no output.
        
        Args:
            self: Description of self.
        """
        @built_workflow
        async def silent_wf(x: int) -> None:
            """Implements silent wf.
            
            Args:
                x: Description of x.
            """
            pass  # returns None — no output emitted

        result = await silent_wf.run(5)
        assert result.get_outputs() == []

    async def test_return_value_auto_yields_output(self):
        """Returning a non-None value automatically emits it as an output."""

        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 3

        result = await wf.run(5)
        assert result.get_outputs() == [15]

    async def test_step_called_multiple_times(self):
        """Validates behavior for step called multiple times.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a = await add_one(x)
            b = await add_one(a)
            return await add_one(b)

        result = await wf.run(0)
        assert result.get_outputs() == [3]  # 0+1+1+1

        # Should have 3 invoked and 3 completed events for add_one
        invoked = [e for e in result if e.type == "executor_invoked"]
        completed = [e for e in result if e.type == "executor_completed"]
        assert len(invoked) == 3
        assert len(completed) == 3


# ---------------------------------------------------------------------------
# Recovery after errors
# ---------------------------------------------------------------------------


class TestRecoveryAfterErrors:
    """Represents the TestRecoveryAfterErrors type and related behavior.
    """
    async def test_run_after_failure_is_allowed(self):
        """Validates behavior for run after failure is allowed.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            if x == 1:
                raise RuntimeError("boom")
            return x

        with pytest.raises(RuntimeError, match="boom"):
            await wf.run(1)

        # Must be able to run again after the failure
        result = await wf.run(2)
        assert result.get_outputs() == [2]

    async def test_step_sync_function_raises(self):
        """Validates behavior for step sync function raises.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        with pytest.raises(TypeError, match="async functions"):

            @step  # type: ignore[arg-type, call-overload]  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
            def not_async(x: int) -> int:  # pyright: ignore[reportUnusedFunction]
                """Implements not async.
                
                Args:
                    x: Description of x.
                
                Returns:
                    Description of the return value.
                """
                return x


# ---------------------------------------------------------------------------
# WorkflowInterrupted is BaseException
# ---------------------------------------------------------------------------


class TestWorkflowInterruptedIsBaseException:
    """Represents the TestWorkflowInterruptedIsBaseException type and related behavior.
    """
    async def test_except_exception_does_not_catch_interrupt(self):
        """User code with ``except Exception`` should not catch WorkflowInterrupted."""
        caught = False

        @built_workflow
        async def wf(x: int, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            nonlocal caught
            try:
                return await ctx.request_info("need review", response_type=str, request_id="r1")
            except Exception:
                # This should NOT catch WorkflowInterrupted
                caught = True
                return "caught!"

        result = await wf.run("data")
        # Should have a pending request, NOT "caught!"
        assert result.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        assert result.get_outputs() == []
        assert caught is False


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------


class TestCheckpointValidation:
    """Represents the TestCheckpointValidation type and related behavior.
    """
    async def test_checkpoint_signature_mismatch_raises(self):
        """Validates behavior for checkpoint signature mismatch raises.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        from agent_framework import WorkflowCheckpoint

        storage = InMemoryCheckpointStorage()

        @built_workflow(name="my_wf", checkpoint_storage=storage)
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        # Manually create a checkpoint with a different signature hash
        bad_checkpoint = WorkflowCheckpoint(
            workflow_name="my_wf",
            graph_signature_hash="totally_different_hash",
            state={"_step_cache": {}, "_original_message": 1},
        )
        ckpt_id = await storage.save(bad_checkpoint)

        # Should fail due to hash mismatch
        with pytest.raises(ValueError, match="not compatible"):
            await wf.run(checkpoint_id=ckpt_id)

    async def test_import_step_cache_malformed_key(self):
        """Validates behavior for import step cache malformed key.
        
        Args:
            self: Description of self.
        """
        ctx = _RunContext("test")
        with pytest.raises(ValueError, match="Corrupted step cache"):
            ctx._import_step_cache({"invalid_key_no_separator": 42})  # pyright: ignore[reportPrivateUsage]

    async def test_import_step_cache_non_integer_index(self):
        """Validates behavior for import step cache non integer index.
        
        Args:
            self: Description of self.
        """
        ctx = _RunContext("test")
        with pytest.raises(ValueError, match="Corrupted step cache"):
            ctx._import_step_cache({"step_name::abc": 42})  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# executor_bypassed event on replay (review comment #3)
# ---------------------------------------------------------------------------


class TestExecutorBypassed:
    """Represents the TestExecutorBypassed type and related behavior.
    """
    async def test_cached_step_emits_bypassed_event(self):
        """When a step replays from cache, it should emit executor_bypassed."""
        storage = InMemoryCheckpointStorage()
        call_count = 0

        @step
        async def tracked(x: int) -> int:
            """Implements tracked.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            nonlocal call_count
            call_count += 1
            return x + 1

        @built_workflow(checkpoint_storage=storage)
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await tracked(x)

        # First run — live execution
        result1 = await wf.run(5)
        assert result1.get_outputs() == [6]
        assert call_count == 1

        event_types1 = [e.type for e in result1]
        assert "executor_invoked" in event_types1
        assert "executor_completed" in event_types1
        assert "executor_bypassed" not in event_types1

        # Restore from checkpoint — cached replay
        ckpt_id = (await storage.list_checkpoints(workflow_name="wf"))[-1].checkpoint_id
        result2 = await wf.run(checkpoint_id=ckpt_id)
        assert result2.get_outputs() == [6]
        assert call_count == 1  # not called again

        event_types2 = [e.type for e in result2]
        assert "executor_bypassed" in event_types2
        # Should NOT have the live-execution pair
        assert "executor_invoked" not in event_types2
        assert "executor_completed" not in event_types2

    async def test_bypassed_event_carries_cached_data(self):
        """Validates behavior for bypassed event carries cached data.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()

        @step
        async def compute(x: int) -> int:
            """Implements compute.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 10

        @built_workflow(checkpoint_storage=storage)
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await compute(x)

        await wf.run(3)
        ckpt_id = (await storage.list_checkpoints(workflow_name="wf"))[-1].checkpoint_id

        result = await wf.run(checkpoint_id=ckpt_id)
        bypassed = [e for e in result if e.type == "executor_bypassed"]
        assert len(bypassed) == 1
        assert bypassed[0].executor_id == "compute"
        assert bypassed[0].data == 30


# ---------------------------------------------------------------------------
# request_info inside @step (review comment #1)
# ---------------------------------------------------------------------------


class TestRequestInfoInStep:
    """Represents the TestRequestInfoInStep type and related behavior.
    """
    async def test_step_with_run_context_injection(self):
        """A @step function with a RunContext parameter gets it auto-injected."""

        @step
        async def review_step(doc: str, ctx: RunContext) -> str:
            """Implements review step.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"draft": doc}, response_type=str, request_id="s1")
            return f"reviewed: {feedback}"

        @built_workflow
        async def wf(doc: str) -> str:
            """Implements wf.
            
            Args:
                doc: Description of doc.
            
            Returns:
                Description of the return value.
            """
            return await review_step(doc)

        # Phase 1: should interrupt
        result1 = await wf.run("my doc")
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        assert len(result1.get_request_info_events()) == 1
        assert result1.get_request_info_events()[0].request_id == "s1"

        # Phase 2: resume
        result2 = await wf.run(responses={"s1": "LGTM"})
        assert result2.get_outputs() == ["reviewed: LGTM"]

    async def test_step_works_outside_workflow_with_explicit_ctx(self):
        """Outside a workflow, the step is transparent — caller provides ctx."""

        @step
        async def needs_ctx(data: str, ctx: RunContext) -> str:
            """Implements needs ctx.
            
            Args:
                data: Description of data.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            val = ctx.get_state("key", "default")
            return f"{data}:{val}"

        # Outside a workflow, pass through directly — caller supplies ctx
        ctx = RunContext("test")
        ctx.set_state("key", "hello")
        result = await needs_ctx("data", ctx)
        assert result == "data:hello"

    async def test_step_injects_ctx_before_user_positional_parameters(self):
        """RunContext injection should not conflict when ctx is the first step parameter."""

        @step
        async def needs_ctx_first(ctx: RunContext, data: str) -> str:
            """Implements needs ctx first.
            
            Args:
                ctx: Description of ctx.
                data: Description of data.
            
            Returns:
                Description of the return value.
            """
            ctx.set_state("seen", data)
            return f"{data}:{ctx.get_state('seen')}"

        @built_workflow
        async def wf(data: str) -> str:
            """Implements wf.
            
            Args:
                data: Description of data.
            
            Returns:
                Description of the return value.
            """
            return await needs_ctx_first(data)

        result = await wf.run("draft")

        assert result.get_outputs() == ["draft:draft"]

    async def test_get_run_context_inside_workflow(self):
        """get_run_context() returns the active RunContext inside a workflow."""
        from agent_framework import get_run_context

        captured_ctx = None

        @step
        async def capture_ctx(x: int) -> int:
            """Implements capture ctx.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            nonlocal captured_ctx
            captured_ctx = get_run_context()  # type: ignore[assignment]
            return x

        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await capture_ctx(x)

        await wf.run(1)
        assert captured_ctx is not None
        assert isinstance(captured_ctx, RunContext)

    async def test_get_run_context_outside_workflow(self):
        """get_run_context() returns None outside a workflow."""
        from agent_framework import get_run_context

        assert get_run_context() is None


# ---------------------------------------------------------------------------
# None response handling (review comment #2)
# ---------------------------------------------------------------------------


class TestNoneResponseHandling:
    """Represents the TestNoneResponseHandling type and related behavior.
    """
    async def test_none_response_logs_warning(self):
        """Providing None as a response value should log a warning."""

        @built_workflow
        async def wf(doc: str, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            val = await ctx.request_info("need input", response_type=str, request_id="r1")
            return f"got: {val}"

        # Phase 1
        await wf.run("start")

        # Phase 2: resume with None response — should warn but still work
        with caplog_context(logging.getLogger("agent_framework._workflows._functional")) as logs:
            result = await wf.run(responses={"r1": None})

        assert result.get_outputs() == ["got: None"]
        assert any("None" in msg and "r1" in msg for msg in logs)

    async def test_none_response_is_returned(self):
        """None is a valid (if discouraged) response value."""

        @built_workflow
        async def wf(x: int, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            val = await ctx.request_info("need data", response_type=str, request_id="r1")
            return f"value={val}"

        await wf.run(1)
        result = await wf.run(responses={"r1": None})
        assert result.get_outputs() == ["value=None"]


# Helper for capturing log messages


@contextmanager
def caplog_context(target_logger: logging.Logger) -> Iterator[list[str]]:
    """Capture log messages from a specific logger."""
    messages: list[str] = []

    class _Handler(logging.Handler):
        """Represents the _Handler type and related behavior.
        """
        def emit(self, record: logging.LogRecord) -> None:
            """Implements emit.
            
            Args:
                self: Description of self.
                record: Description of record.
            """
            messages.append(self.format(record))

    handler = _Handler()
    handler.setLevel(logging.WARNING)
    target_logger.addHandler(handler)
    try:
        yield messages
    finally:
        target_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Combined regression tests (cross-cutting review comments #1, #2, #3)
# ---------------------------------------------------------------------------


class TestHITLInStepWithCaching:
    """Regression tests: request_info inside @step combined with caching and bypass."""

    async def test_preceding_step_bypassed_on_hitl_resume(self):
        """When a step after a completed step calls request_info and interrupts,
        resuming should bypass the first step (cached) and re-execute the HITL step."""
        call_count_a = 0

        @step
        async def step_a(x: int) -> int:
            """Implements step a.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            nonlocal call_count_a
            call_count_a += 1
            return x + 1

        @step
        async def step_b(val: int, ctx: RunContext) -> str:
            """Implements step b.
            
            Args:
                val: Description of val.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"val": val}, response_type=str, request_id="r1")
            return f"{val}:{feedback}"

        @built_workflow
        async def wf(x: int) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            a = await step_a(x)
            return await step_b(a)

        # Phase 1: step_a completes, step_b interrupts
        result1 = await wf.run(5)
        assert call_count_a == 1
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # Phase 2: resume — step_a should be bypassed, step_b re-executes
        result2 = await wf.run(responses={"r1": "ok"})
        assert call_count_a == 1  # step_a not called again
        assert result2.get_outputs() == ["6:ok"]

        event_types = [e.type for e in result2]
        assert "executor_bypassed" in event_types

    async def test_hitl_step_with_checkpoint_full_lifecycle(self):
        """Full lifecycle: run -> interrupt -> resume -> checkpoint restore -> all bypassed."""
        storage = InMemoryCheckpointStorage()

        @step
        async def compute(x: int) -> int:
            """Implements compute.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 10

        @step
        async def review(val: int, ctx: RunContext) -> str:
            """Implements review.
            
            Args:
                val: Description of val.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            feedback = await ctx.request_info({"val": val}, response_type=str, request_id="rev")
            return f"reviewed({val}):{feedback}"

        @built_workflow(checkpoint_storage=storage)
        async def wf(x: int) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            v = await compute(x)
            return await review(v)

        # Phase 1: interrupt
        result1 = await wf.run(3)
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS

        # Phase 2: resume
        result2 = await wf.run(responses={"rev": "LGTM"})
        assert result2.get_outputs() == ["reviewed(30):LGTM"]

        # Phase 3: restore from latest checkpoint -- both steps should be bypassed
        ckpt_id = (await storage.list_checkpoints(workflow_name="wf"))[-1].checkpoint_id
        result3 = await wf.run(checkpoint_id=ckpt_id)
        assert result3.get_outputs() == ["reviewed(30):LGTM"]

        event_types3 = [e.type for e in result3]
        bypassed = [e for e in result3 if e.type == "executor_bypassed"]
        assert len(bypassed) == 2
        assert "executor_invoked" not in event_types3

    async def test_none_response_in_step_request_info(self):
        """None response inside a @step request_info should warn and return None."""

        @step
        async def needs_feedback(doc: str, ctx: RunContext) -> str:
            """Implements needs feedback.
            
            Args:
                doc: Description of doc.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            val = await ctx.request_info({"doc": doc}, response_type=str, request_id="r1")
            return f"got:{val}"

        @built_workflow
        async def wf(doc: str) -> str:
            """Implements wf.
            
            Args:
                doc: Description of doc.
            
            Returns:
                Description of the return value.
            """
            return await needs_feedback(doc)

        await wf.run("draft")

        with caplog_context(logging.getLogger("agent_framework._workflows._functional")) as logs:
            result = await wf.run(responses={"r1": None})

        assert result.get_outputs() == ["got:None"]
        assert any("None" in msg and "r1" in msg for msg in logs)

    async def test_step_hitl_does_not_emit_executor_failed(self):
        """WorkflowInterrupted from request_info inside a step should NOT emit executor_failed."""

        @step
        async def hitl_step(x: int, ctx: RunContext) -> str:
            """Implements hitl step.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return await ctx.request_info("need data", response_type=str, request_id="r1")

        @built_workflow
        async def wf(x: int) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return await hitl_step(x)

        result = await wf.run(1)
        event_types = [e.type for e in result]
        assert "executor_failed" not in event_types
        assert result.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS


# ---------------------------------------------------------------------------
# Regression tests for ultrareview findings
# ---------------------------------------------------------------------------


class TestDeterministicAutoRequestId:
    """Regression for bug_001: auto-generated request_info ids must be stable across replay."""

    async def test_auto_request_id_roundtrips_on_resume(self):
        """Validates behavior for auto request id roundtrips on resume.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int, ctx: RunContext) -> str:
            # No request_id — framework must generate a deterministic one
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            val = await ctx.request_info("need data", response_type=str)
            return f"got:{val}"

        result1 = await wf.run(1)
        assert result1.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS
        requests = result1.get_request_info_events()
        assert len(requests) == 1
        rid = requests[0].request_id
        assert rid  # non-empty

        # Resume with the id the caller just received.
        result2 = await wf.run(responses={rid: "hello"})
        assert result2.get_final_state() == WorkflowRunState.IDLE
        assert result2.get_outputs() == ["got:hello"]

    async def test_multiple_auto_ids_are_distinct_and_stable(self):
        """Validates behavior for multiple auto ids are distinct and stable.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            a = await ctx.request_info("first", response_type=str)
            b = await ctx.request_info("second", response_type=str)
            return f"{a}/{b}"

        r1 = await wf.run(1)
        rid1 = r1.get_request_info_events()[0].request_id
        r2 = await wf.run(responses={rid1: "A"})
        rid2 = r2.get_request_info_events()[0].request_id
        assert rid1 != rid2
        r3 = await wf.run(responses={rid1: "A", rid2: "B"})
        assert r3.get_outputs() == ["A/B"]

    async def test_cached_step_advances_auto_request_id_counter(self):
        """Validates behavior for cached step advances auto request id counter.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        call_count = 0

        @step
        async def first_review(value: int, ctx: RunContext) -> str:
            """Implements first review.
            
            Args:
                value: Description of value.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            nonlocal call_count
            call_count += 1
            return await ctx.request_info({"step": "first", "value": value}, response_type=str)

        @step
        async def second_review(value: int, ctx: RunContext) -> str:
            """Implements second review.
            
            Args:
                value: Description of value.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return await ctx.request_info({"step": "second", "value": value}, response_type=str)

        @built_workflow
        async def wf(value: int) -> str:
            """Implements wf.
            
            Args:
                value: Description of value.
            
            Returns:
                Description of the return value.
            """
            first = await first_review(value)
            second = await second_review(value)
            return f"{first}/{second}"

        first_run = await wf.run(1)
        first_request_id = first_run.get_request_info_events()[0].request_id
        assert first_request_id == "auto::0"

        second_run = await wf.run(responses={first_request_id: "A"})
        second_request_id = second_run.get_request_info_events()[0].request_id
        assert second_request_id == "auto::1"
        completed_call_count = call_count

        final_run = await wf.run(responses={first_request_id: "A", second_request_id: "B"})

        assert call_count == completed_call_count
        assert final_run.get_outputs() == ["A/B"]


class TestPendingRequestsPruned:
    """Regression for bug_007: resolved requests must be pruned from _pending_requests."""

    async def test_final_checkpoint_no_longer_claims_resolved_requests_pending(self):
        """Validates behavior for final checkpoint no longer claims resolved requests pending.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        storage = InMemoryCheckpointStorage()

        @built_workflow(checkpoint_storage=storage)
        async def wf(x: int, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            a = await ctx.request_info("q1", response_type=str, request_id="r1")
            b = await ctx.request_info("q2", response_type=str, request_id="r2")
            return f"{a}/{b}"

        await wf.run(1)
        await wf.run(responses={"r1": "A"})
        result = await wf.run(responses={"r1": "A", "r2": "B"})
        assert result.get_final_state() == WorkflowRunState.IDLE
        # Latest checkpoint must show no pending requests.
        checkpoints = await storage.list_checkpoints(workflow_name="wf")
        assert checkpoints, "expected at least one checkpoint to have been saved"
        final = checkpoints[-1]
        assert final.pending_request_info_events == {}


class TestArityValidation:
    """Regression for merged_bug_003: validate workflow signature arity."""

    def test_multi_non_ctx_param_rejected_at_decoration(self):
        """Validates behavior for multi non ctx param rejected at decoration.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        with pytest.raises(ValueError, match="multiple non-RunContext parameters"):

            @workflow
            async def wf(a: str, b: str, ctx: RunContext) -> str:
                """Implements wf.
                
                Args:
                    a: Description of a.
                    b: Description of b.
                    ctx: Description of ctx.
                
                Returns:
                    Description of the return value.
                """
                return f"{a}+{b}"

    async def test_ctx_only_workflow_with_message_raises_clear_error(self):
        """Validates behavior for ctx only workflow with message raises clear error.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return "no message used"

        with pytest.raises(ValueError, match="no non-RunContext parameter"):
            await wf.run("important input")

    def test_ctx_only_workflow_decoration_succeeds(self):
        # Decoration must not raise even though the workflow has no
        # message-receiving parameter.  (Running it without a message still
        # requires providing responses or a checkpoint_id — that's
        # _validate_run_params's job, not ours.)
        """Validates behavior for ctx only workflow decoration succeeds.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return "ok"

        assert wf is not None


class TestStaleResponsesRejected:
    """Regression for bug_014: stale responses after clean completion must be rejected."""

    async def test_responses_after_clean_completion_raise(self):
        """Validates behavior for responses after clean completion raise.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 2

        await wf.run(5)  # clean completion, no pending requests
        with pytest.raises(ValueError, match="no pending request_info"):
            await wf.run(responses={"stale": "x"})

    async def test_responses_mismatched_key_raises(self):
        """Validates behavior for responses mismatched key raises.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            return await ctx.request_info("q", response_type=str, request_id="r1")

        await wf.run(1)  # interrupts with r1 pending
        with pytest.raises(ValueError, match="do not answer"):
            await wf.run(responses={"definitely_not_r1": "x"})


class TestReservedStateKeys:
    """Regression for bug_017: set_state must reject underscore-prefixed keys."""

    async def test_underscore_key_rejected(self):
        """Validates behavior for underscore key rejected.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int, ctx: RunContext) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            ctx.set_state("_private", "user value")
            return x

        with pytest.raises(ValueError, match="reserved for framework"):
            await wf.run(1)

    async def test_normal_key_still_works(self):
        """Validates behavior for normal key still works.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: int, ctx: RunContext) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            ctx.set_state("normal_key", "v")
            assert ctx.get_state("normal_key") == "v"
            return x

        r = await wf.run(1)
        assert r.get_outputs() == [1]


class TestDeepcopyOnCacheHit:
    """Regression for bug_002: cache hits must not deepcopy args."""

    async def test_step_with_non_deepcopyable_arg_replays(self):
        """Validates behavior for step with non deepcopyable arg replays.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        import threading

        @step
        async def takes_lock(lock: threading.Lock, n: int) -> int:
            """Implements takes lock.
            
            Args:
                lock: Description of lock.
                n: Description of n.
            
            Returns:
                Description of the return value.
            """
            return n + 1

        @built_workflow
        async def wf(x: int) -> int:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            lock = threading.Lock()
            return await takes_lock(lock, x)

        # First run — must succeed despite threading.Lock not being deepcopyable
        # (deepcopy now wrapped in try/except, falls back to live reference for
        # the invocation_data event only).
        r1 = await wf.run(5)
        assert r1.get_outputs() == [6]


class TestStepDiscoveryAttributeAccess:
    """Regression for bug_008: checkpoint hash must differ when function body changes."""

    async def test_signature_hash_changes_when_function_body_changes(self):
        """Validates behavior for signature hash changes when function body changes.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf_a(x: int) -> int:
            """Implements wf a.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x + 1

        @built_workflow(name="wf_b")
        async def wf_b(x: int) -> int:
            """Implements wf b.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x * 100

        # Two different function bodies -> different hashes even though the
        # static step-name scan would produce the same empty list.
        assert wf_a.graph_signature_hash != wf_b.graph_signature_hash


class TestAsAgentSignatureParity:
    """Regression for bug_015: as_agent signature must accept description/context_providers."""

    async def test_as_agent_accepts_description_override(self):
        """Validates behavior for as agent accepts description override.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow(description="workflow level")
        async def wf(x: str) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x.upper()

        agent = wf.as_agent(name="a", description="agent level")
        assert agent.description == "agent level"

    async def test_as_agent_accepts_context_providers_kwarg(self):
        """Validates behavior for as agent accepts context providers kwarg.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: str) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        providers = [object()]  # opaque placeholder; must be stored without error
        agent = wf.as_agent(context_providers=providers)
        assert list(agent.context_providers or []) == providers

    async def test_as_agent_description_defaults_to_workflow_description(self):
        """Validates behavior for as agent description defaults to workflow description.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow(description="from workflow")
        async def wf(x: str) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
            
            Returns:
                Description of the return value.
            """
            return x

        agent = wf.as_agent()
        assert agent.description == "from workflow"


class TestFunctionalWorkflowAgentHITL:
    """Regression for bug_013: .as_agent() must surface request_info events."""

    async def test_request_info_surfaces_as_function_approval_request(self):
        """Validates behavior for request info surfaces as function approval request.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: str, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            answer = await ctx.request_info({"need": x}, response_type=str, request_id="rid-1")
            return f"got:{answer}"

        agent = wf.as_agent()
        response = await agent.run("topic")

        # Agent must expose the pending request_id.
        assert "rid-1" in agent.pending_requests

        # Response must contain at least one content item whose type is
        # function_approval_request (or equivalent).
        approval_found = False
        for message in response.messages:
            for content in message.contents:
                if getattr(content, "type", None) == "function_approval_request":
                    approval_found = True
                    break
        assert approval_found, "expected FunctionApprovalRequestContent in agent response"

    async def test_request_info_dataclass_arguments_are_serialized_for_agent(self):
        """Validates behavior for request info dataclass arguments are serialized for agent.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @dataclass
        class HandoffRequest:
            """Represents the HandoffRequest type and related behavior.
            """
            target_agent: str
            reason: str

        @built_workflow
        async def wf(x: str, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            answer = await ctx.request_info(
                HandoffRequest(target_agent=x, reason="overflow"),
                response_type=str,
                request_id="rid-1",
            )
            return f"got:{answer}"

        agent = wf.as_agent()
        response = await agent.run("helper")

        function_call_arguments = None
        for message in response.messages:
            for content in message.contents:
                if getattr(content, "type", None) == "function_approval_request" and content.function_call is not None:
                    function_call_arguments = content.function_call.arguments
                    break

        assert function_call_arguments == {
            "request_id": "rid-1",
            "data": {"target_agent": "helper", "reason": "overflow"},
        }
        assert json.loads(json.dumps(function_call_arguments)) == function_call_arguments

    async def test_resume_via_agent_responses_kwarg(self):
        """Validates behavior for resume via agent responses kwarg.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        @built_workflow
        async def wf(x: str, ctx: RunContext) -> str:
            """Implements wf.
            
            Args:
                x: Description of x.
                ctx: Description of ctx.
            
            Returns:
                Description of the return value.
            """
            answer = await ctx.request_info(x, response_type=str, request_id="rid-1")
            return f"got:{answer}"

        agent = wf.as_agent()
        # First phase: suspend
        await agent.run("topic")
        # Second phase: resume via the agent surface
        response = await agent.run(responses={"rid-1": "answered"})
        # Agent's final response should contain the workflow's text output.
        text_blobs: list[str] = []
        for message in response.messages:
            for content in message.contents:
                text = getattr(content, "text", None)
                if text:
                    text_blobs.append(text)
        assert any("got:answered" in t for t in text_blobs)


class TestRunDocstringAllowsResponsesAndCheckpoint:
    """Regression for bug_010: docstring must permit responses+checkpoint_id combo."""

    def test_docstring_says_at_least_one(self):
        """Validates behavior for docstring says at least one.
        
        Args:
            self: Description of self.
        """
        doc = FunctionalWorkflow.run.__doc__ or ""
        assert "At least one" in doc or "at least one" in doc
        assert "Exactly one" not in doc


class TestFunctionalWorkflowExperimentalStage:
    """Tests for the experimental stage annotations applied to functional workflow APIs."""

    def test_public_symbols_are_marked_experimental(self) -> None:
        """Validates behavior for public symbols are marked experimental.
        
        Args:
            self: Description of self.
        """
        symbols = [
            get_run_context,
            RunContext,
            StepWrapper,
            step,
            FunctionalWorkflow,
            FunctionalWorkflowDefinition,
            workflow,
            FunctionalWorkflowAgent,
        ]

        for symbol in symbols:
            assert symbol.__feature_stage__ == "experimental"  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
            assert symbol.__feature_id__ == ExperimentalFeature.FUNCTIONAL_WORKFLOWS.value  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
            assert symbol.__doc__ is not None
            assert ".. warning:: Experimental" in symbol.__doc__
