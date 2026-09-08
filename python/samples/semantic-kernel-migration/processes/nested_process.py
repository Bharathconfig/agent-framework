# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "agent-framework-core",
#     "python-dotenv",
#     "semantic-kernel",
# ]
# ///
# Run with any PEP 723 compatible runner, e.g.:
#   uv run samples/semantic-kernel-migration/processes/nested_process.py

# Copyright (c) Microsoft. All rights reserved.

"""Nested process comparison between Semantic Kernel Process Framework and Agent Framework sub-workflows."""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, cast

######################################################################
# region Agent Framework imports
######################################################################
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowExecutor,
    handler,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field

######################################################################
# region Semantic Kernel imports
######################################################################
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEventVisibility
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_context import KernelProcessStepContext
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.local_runtime.local_kernel_process import start
from semantic_kernel.processes.process_builder import ProcessBuilder
from typing_extensions import Never

######################################################################
# endregion
######################################################################
logging.basicConfig(level=logging.WARNING)

# Load environment variables from .env file
load_dotenv()


class ProcessEvents(Enum):
    """Represents the ProcessEvents type and related behavior.
    """
    START_PROCESS = "StartProcess"
    START_INNER_PROCESS = "StartInnerProcess"
    OUTPUT_READY_PUBLIC = "OutputReadyPublic"
    OUTPUT_READY_INTERNAL = "OutputReadyInternal"


######################################################################
# region Semantic Kernel nested process path
######################################################################


class StepState(BaseModel):
    """Represents the StepState type and related behavior.
    """
    last_message: str | None = None


class EchoStep(KernelProcessStep[None]):
    """Represents the EchoStep type and related behavior.
    """
    ECHO: ClassVar[str] = "echo"

    @kernel_function(name=ECHO)
    async def echo(self, message: str) -> str:
        """Implements echo.
        
        Args:
            self: Description of self.
            message: Description of message.
        
        Returns:
            Description of the return value.
        """
        print(f"[ECHO] {message}")
        return message


class RepeatStep(KernelProcessStep[StepState]):
    """Represents the RepeatStep type and related behavior.
    """
    REPEAT: ClassVar[str] = "repeat"

    state: StepState = Field(default_factory=StepState)

    async def activate(self, state: KernelProcessStepState[StepState]):
        """Implements activate.
        
        Args:
            self: Description of self.
            state: Description of state.
        """
        self.state = state.state

    @kernel_function(name=REPEAT)
    async def repeat(
        self,
        message: str,
        context: KernelProcessStepContext,
        count: int = 2,
    ) -> None:
        """Implements repeat.
        
        Args:
            self: Description of self.
            message: Description of message.
            context: Description of context.
            count: Description of count.
        """
        output = " ".join([message] * count)
        self.state.last_message = output
        print(f"[REPEAT] {output}")

        await context.emit_event(
            process_event=ProcessEvents.OUTPUT_READY_PUBLIC.value,
            data=output,
            visibility=KernelProcessEventVisibility.Public,
        )
        await context.emit_event(
            process_event=ProcessEvents.OUTPUT_READY_INTERNAL.value,
            data=output,
            visibility=KernelProcessEventVisibility.Internal,
        )


def _create_linear_process(name: str) -> ProcessBuilder:
    """Implements  create linear process.
    
    Args:
        name: Description of name.
    
    Returns:
        Description of the return value.
    """
    process_builder = ProcessBuilder(name=name)
    echo_step = process_builder.add_step(step_type=EchoStep)
    repeat_step = process_builder.add_step(step_type=RepeatStep)

    process_builder.on_input_event(event_id=ProcessEvents.START_PROCESS.value).send_event_to(target=echo_step)

    echo_step.on_function_result(function_name=EchoStep.ECHO).send_event_to(
        target=repeat_step,
        parameter_name="message",
    )

    return process_builder


_semantic_kernel = Kernel()


async def run_semantic_kernel_nested_process() -> None:
    """Implements run semantic kernel nested process.
    """
    _semantic_kernel.add_service(OpenAIChatCompletion(service_id="default"))

    process_builder = _create_linear_process("Outer")
    nested_process_step = process_builder.add_step_from_process(_create_linear_process("Inner"))

    process_builder.steps[1].on_event(ProcessEvents.OUTPUT_READY_INTERNAL.value).send_event_to(
        nested_process_step.where_input_event_is(ProcessEvents.START_PROCESS.value)
    )

    kernel_process = process_builder.build()

    process_handle = await start(
        process=kernel_process,
        kernel=_semantic_kernel,
        initial_event=ProcessEvents.START_PROCESS.value,
        data="Test",
    )
    process_info = await process_handle.get_state()

    inner_process: KernelProcess | None = next(
        (s for s in process_info.steps if s.state.name == "Inner"),
        None,
    )
    if inner_process is None:
        raise RuntimeError("Inner process state missing")

    repeat_state: KernelProcessStepState[StepState] | None = next(
        (s.state for s in inner_process.steps if s.state.name == "RepeatStep"),
        None,
    )
    if repeat_state is None or repeat_state.state is None:
        raise RuntimeError("RepeatStep state missing")
    assert repeat_state.state.last_message == "Test Test Test Test"  # nosec


######################################################################
# region Agent Framework nested workflow path
######################################################################


@dataclass
class RepeatPayload:
    """Represents the RepeatPayload type and related behavior.
    """
    message: str
    count: int = 2


class KickoffExecutor(Executor):
    """Represents the KickoffExecutor type and related behavior.
    """
    def __init__(self) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
        """
        super().__init__(id="kickoff")

    @handler
    async def start(self, message: str, ctx: WorkflowContext[RepeatPayload]) -> None:
        """Implements start.
        
        Args:
            self: Description of self.
            message: Description of message.
            ctx: Description of ctx.
        """
        print(f"[OUTER] Start with message: {message}")
        await ctx.send_message(RepeatPayload(message=message, count=2))


class OuterEchoExecutor(Executor):
    """Represents the OuterEchoExecutor type and related behavior.
    """
    def __init__(self) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
        """
        super().__init__(id="outer_echo")

    @handler
    async def echo(self, payload: RepeatPayload, ctx: WorkflowContext[RepeatPayload]) -> None:
        """Implements echo.
        
        Args:
            self: Description of self.
            payload: Description of payload.
            ctx: Description of ctx.
        """
        print(f"[OUTER ECHO] {payload.message}")
        await ctx.send_message(payload)


class OuterRepeatExecutor(Executor):
    """Represents the OuterRepeatExecutor type and related behavior.
    """
    def __init__(self, *, inner_target_id: str) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
            inner_target_id: Description of inner_target_id.
        """
        super().__init__(id="outer_repeat")
        self._inner_target_id = inner_target_id

    @handler
    async def repeat(self, payload: RepeatPayload, ctx: WorkflowContext[RepeatPayload]) -> None:
        """Implements repeat.
        
        Args:
            self: Description of self.
            payload: Description of payload.
            ctx: Description of ctx.
        """
        repeated = " ".join([payload.message] * payload.count)
        print(f"[OUTER REPEAT] {repeated}")
        await ctx.send_message(RepeatPayload(message=repeated, count=2), target_id=self._inner_target_id)


class InnerEchoExecutor(Executor):
    """Represents the InnerEchoExecutor type and related behavior.
    """
    def __init__(self) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
        """
        super().__init__(id="inner_echo")

    @handler
    async def echo(self, payload: RepeatPayload, ctx: WorkflowContext[RepeatPayload]) -> None:
        """Implements echo.
        
        Args:
            self: Description of self.
            payload: Description of payload.
            ctx: Description of ctx.
        """
        print(f"    [INNER ECHO] {payload.message}")
        await ctx.send_message(payload)


class InnerRepeatExecutor(Executor):
    """Represents the InnerRepeatExecutor type and related behavior.
    """
    def __init__(self) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
        """
        super().__init__(id="inner_repeat")

    @handler
    async def repeat(self, payload: RepeatPayload, ctx: WorkflowContext[Never, str]) -> None:
        """Implements repeat.
        
        Args:
            self: Description of self.
            payload: Description of payload.
            ctx: Description of ctx.
        """
        repeated = " ".join([payload.message] * payload.count)
        print(f"    [INNER REPEAT] {repeated}")
        await ctx.yield_output(repeated)


class CollectResultExecutor(Executor):
    """Represents the CollectResultExecutor type and related behavior.
    """
    def __init__(self) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
        """
        super().__init__(id="collector")

    @handler
    async def collect(self, result: str, ctx: WorkflowContext[Never, str]) -> None:
        """Implements collect.
        
        Args:
            self: Description of self.
            result: Description of result.
            ctx: Description of ctx.
        """
        print(f"[COLLECTOR] Final result -> {result}")
        await ctx.yield_output(result)


def _build_inner_workflow() -> WorkflowExecutor:
    """Implements  build inner workflow.
    
    Returns:
        Description of the return value.
    """
    inner_echo = InnerEchoExecutor()
    inner_repeat = InnerRepeatExecutor()

    inner_workflow = WorkflowBuilder(start_executor=inner_echo).add_edge(inner_echo, inner_repeat).build()

    return WorkflowExecutor(inner_workflow, id="inner_workflow")


async def run_agent_framework_nested_workflow(initial_message: str) -> Sequence[str]:
    """Implements run agent framework nested workflow.
    
    Args:
        initial_message: Description of initial_message.
    
    Returns:
        Description of the return value.
    """
    inner_executor = _build_inner_workflow()

    kickoff = KickoffExecutor()
    outer_echo = OuterEchoExecutor()
    outer_repeat = OuterRepeatExecutor(inner_target_id=inner_executor.id)
    collector = CollectResultExecutor()

    outer_workflow = (
        WorkflowBuilder(start_executor=kickoff)
        .add_edge(kickoff, outer_echo)
        .add_edge(outer_echo, outer_repeat)
        .add_edge(outer_repeat, inner_executor)
        .add_edge(inner_executor, collector)
        .build()
    )

    results: list[str] = []
    async for event in outer_workflow.run(initial_message, stream=True):
        if event.type == "output":
            results.append(cast(str, event.data))

    return results


######################################################################
# endregion
######################################################################


async def main() -> None:
    """Implements main.
    """
    print("===== Agent Framework Nested Workflow =====")
    af_results = await run_agent_framework_nested_workflow("Test")
    for index, value in enumerate(af_results, start=1):
        print(f"Result {index}: {value}")

    print("\n===== Semantic Kernel Nested Process =====")
    await run_semantic_kernel_nested_process()


if __name__ == "__main__":
    asyncio.run(main())
