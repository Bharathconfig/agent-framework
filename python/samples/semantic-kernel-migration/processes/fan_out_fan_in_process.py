# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "agent-framework-core",
#     "python-dotenv",
#     "semantic-kernel",
# ]
# ///
# Run with any PEP 723 compatible runner, e.g.:
#   uv run samples/semantic-kernel-migration/processes/fan_out_fan_in_process.py

# Copyright (c) Microsoft. All rights reserved.

"""Side-by-side sample comparing Semantic Kernel Process Framework and Agent Framework workflows."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, cast

######################################################################
# region Agent Framework imports
######################################################################
from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from dotenv import load_dotenv
from pydantic import BaseModel, Field

######################################################################
# region Semantic Kernel imports
######################################################################
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_context import KernelProcessStepContext
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.process_builder import ProcessBuilder

if TYPE_CHECKING:
    from semantic_kernel.processes.kernel_process import KernelProcess
    from semantic_kernel.processes.local_runtime.local_kernel_process import LocalKernelProcessContext

# Load environment variables from .env file
load_dotenv()


async def _start_local_kernel_process(
    *,
    process: "KernelProcess",
    kernel: Kernel,
    initial_event: KernelProcessEvent | str | Enum,
    **kwargs: object,
) -> "LocalKernelProcessContext":
    """Implements  start local kernel process.
    
    Args:
        process: Description of process.
        kernel: Description of kernel.
        initial_event: Description of initial_event.
        **kwargs: Description of kwargs.
    
    Returns:
        Description of the return value.
    """
    from semantic_kernel.processes.local_runtime.local_kernel_process import start as start_local_kernel_process

    return await start_local_kernel_process(
        process=process,
        kernel=kernel,
        initial_event=initial_event,
        **kwargs,
    )


logging.basicConfig(level=logging.WARNING)


class CommonEvents(Enum):
    """Common events for both samples."""

    USER_INPUT_RECEIVED = "UserInputReceived"
    COMPLETION_RESPONSE_GENERATED = "CompletionResponseGenerated"
    WELCOME_DONE = "WelcomeDone"
    A_STEP_DONE = "AStepDone"
    B_STEP_DONE = "BStepDone"
    C_STEP_DONE = "CStepDone"
    START_A_REQUESTED = "StartARequested"
    START_B_REQUESTED = "StartBRequested"
    EXIT_REQUESTED = "ExitRequested"
    START_PROCESS = "StartProcess"


######################################################################
# region Semantic Kernel Process Framework path
######################################################################


class KickOffStep(KernelProcessStep[None]):
    """Represents the KickOffStep type and related behavior.
    """
    KICK_OFF_FUNCTION: ClassVar[str] = "kick_off"

    @kernel_function(name=KICK_OFF_FUNCTION)
    async def print_welcome_message(self, context: KernelProcessStepContext):
        """Implements print welcome message.
        
        Args:
            self: Description of self.
            context: Description of context.
        """
        await context.emit_event(process_event=CommonEvents.START_A_REQUESTED, data="Get Going A")
        await context.emit_event(process_event=CommonEvents.START_B_REQUESTED, data="Get Going B")


class AStep(KernelProcessStep[None]):
    """Represents the AStep type and related behavior.
    """
    @kernel_function()
    async def do_it(self, context: KernelProcessStepContext):
        """Implements do it.
        
        Args:
            self: Description of self.
            context: Description of context.
        """
        await asyncio.sleep(1)
        await context.emit_event(process_event=CommonEvents.A_STEP_DONE.value, data="I did A")


class BStep(KernelProcessStep[None]):
    """Represents the BStep type and related behavior.
    """
    @kernel_function()
    async def do_it(self, context: KernelProcessStepContext):
        """Implements do it.
        
        Args:
            self: Description of self.
            context: Description of context.
        """
        await asyncio.sleep(2)
        await context.emit_event(process_event=CommonEvents.B_STEP_DONE.value, data="I did B")


class CStepState(BaseModel):
    """Represents the CStepState type and related behavior.
    """
    current_cycle: int = 0


class CStep(KernelProcessStep[CStepState]):
    """Represents the CStep type and related behavior.
    """
    state: CStepState = Field(default_factory=CStepState)

    async def activate(self, state: KernelProcessStepState[CStepState]):
        """Implements activate.
        
        Args:
            self: Description of self.
            state: Description of state.
        """
        self.state = state.state

    @kernel_function()
    async def do_it(self, context: KernelProcessStepContext, astepdata: str, bstepdata: str):
        """Implements do it.
        
        Args:
            self: Description of self.
            context: Description of context.
            astepdata: Description of astepdata.
            bstepdata: Description of bstepdata.
        """
        self.state.current_cycle += 1
        print(f"CStep Current Cycle: {self.state.current_cycle}")
        if self.state.current_cycle == 3:
            print("CStep Exit Requested")
            await context.emit_event(process_event=CommonEvents.EXIT_REQUESTED.value)
            return
        await context.emit_event(process_event=CommonEvents.C_STEP_DONE.value)


kernel = Kernel()


async def run_semantic_kernel_process_example() -> None:
    """Implements run semantic kernel process example.
    """
    kernel.add_service(OpenAIChatCompletion(service_id="default"))

    process = ProcessBuilder(name="Process Framework Sample")

    kickoff_step = process.add_step(step_type=KickOffStep)
    step_a = process.add_step(step_type=AStep)
    step_b = process.add_step(step_type=BStep)
    step_c = process.add_step(step_type=CStep)

    process.on_input_event(event_id=CommonEvents.START_PROCESS.value).send_event_to(target=kickoff_step)

    kickoff_step.on_event(event_id=CommonEvents.START_A_REQUESTED.value).send_event_to(target=step_a)
    kickoff_step.on_event(event_id=CommonEvents.START_B_REQUESTED.value).send_event_to(target=step_b)
    step_a.on_event(event_id=CommonEvents.A_STEP_DONE.value).send_event_to(target=step_c, parameter_name="astepdata")
    step_b.on_event(event_id=CommonEvents.B_STEP_DONE.value).send_event_to(target=step_c, parameter_name="bstepdata")
    step_c.on_event(event_id=CommonEvents.C_STEP_DONE.value).send_event_to(target=kickoff_step)
    step_c.on_event(event_id=CommonEvents.EXIT_REQUESTED.value).stop_process()

    kernel_process: "KernelProcess" = process.build()

    async with await _start_local_kernel_process(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id=CommonEvents.START_PROCESS.value, data="Initial"),
    ) as process_context:
        process_state = await process_context.get_state()
        c_step_state: KernelProcessStepState[CStepState] | None = next(
            (s.state for s in process_state.steps if s.state.name == "CStep"),
            None,
        )
        if c_step_state is None or c_step_state.state is None:
            raise RuntimeError("CStep state unavailable")
        assert c_step_state.state.current_cycle == 3  # nosec
        print(f"Final State Check: CStepState current cycle: {c_step_state.state.current_cycle}")


######################################################################
# region Agent Framework workflow path
######################################################################


@dataclass
class StepResult:
    """Represents the StepResult type and related behavior.
    """
    origin: str
    cycle: int
    data: str


class KickOffExecutor(Executor):
    """Represents the KickOffExecutor type and related behavior.
    """
    def __init__(self, *, id: str = "kickoff") -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
            id: Description of id.
        """
        super().__init__(id=id)
        self._next_cycle = 0

    @handler
    async def handle(self, event: CommonEvents, ctx: WorkflowContext[int]) -> None:
        """Implements handle.
        
        Args:
            self: Description of self.
            event: Description of event.
            ctx: Description of ctx.
        """
        if event not in {CommonEvents.START_PROCESS, CommonEvents.C_STEP_DONE}:
            return
        self._next_cycle += 1
        await ctx.send_message(self._next_cycle)


class DelayedStepExecutor(Executor):
    """Represents the DelayedStepExecutor type and related behavior.
    """
    def __init__(self, *, name: str, delay_seconds: float) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
            name: Description of name.
            delay_seconds: Description of delay_seconds.
        """
        super().__init__(id=name)
        self._delay = delay_seconds
        self._name = name

    @handler
    async def handle(self, cycle: int, ctx: WorkflowContext[StepResult]) -> None:
        """Implements handle.
        
        Args:
            self: Description of self.
            cycle: Description of cycle.
            ctx: Description of ctx.
        """
        await asyncio.sleep(self._delay)
        await ctx.send_message(StepResult(origin=self._name, cycle=cycle, data=f"I did {self._name.upper()[-1]}"))


class FanInExecutor(Executor):
    """Represents the FanInExecutor type and related behavior.
    """
    def __init__(self, *, required_cycles: int = 3, id: str = "fanin") -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
            required_cycles: Description of required_cycles.
            id: Description of id.
        """
        super().__init__(id=id)
        self._completed_cycles = 0
        self._required_cycles = required_cycles

    @handler
    async def handle(self, results: list[StepResult], ctx: WorkflowContext[CommonEvents, str]) -> None:
        """Implements handle.
        
        Args:
            self: Description of self.
            results: Description of results.
            ctx: Description of ctx.
        """
        if not results:
            return
        cycle_number = results[0].cycle
        summary = ", ".join(f"{r.origin}: {r.data}" for r in results)
        print(f"Cycle {cycle_number} aggregate -> {summary}")

        self._completed_cycles += 1
        if self._completed_cycles >= self._required_cycles:
            await ctx.yield_output(f"Completed {self._completed_cycles} cycles")
            return

        await ctx.send_message(CommonEvents.C_STEP_DONE)


async def run_agent_framework_workflow_example() -> str | None:
    """Implements run agent framework workflow example.
    
    Returns:
        Description of the return value.
    """
    kickoff = KickOffExecutor()
    step_a = DelayedStepExecutor(name="step_a", delay_seconds=1)
    step_b = DelayedStepExecutor(name="step_b", delay_seconds=2)
    aggregate = FanInExecutor(required_cycles=3)

    workflow = (
        WorkflowBuilder(start_executor=kickoff)
        .add_edge(kickoff, step_a)
        .add_edge(kickoff, step_b)
        .add_fan_in_edges([step_a, step_b], aggregate)
        .add_edge(aggregate, kickoff)
        .build()
    )

    final_text: str | None = None
    async for event in workflow.run(CommonEvents.START_PROCESS, stream=True):
        if event.type == "output":
            final_text = cast(str, event.data)

    return final_text


async def main() -> None:
    """Implements main.
    """
    print("===== Agent Framework Workflow =====")
    af_result = await run_agent_framework_workflow_example()
    if af_result:
        print(af_result)
    else:
        print("No Agent Framework output.")

    print("===== Semantic Kernel Process Framework =====")
    await run_semantic_kernel_process_example()


if __name__ == "__main__":
    asyncio.run(main())
