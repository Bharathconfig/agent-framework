# Copyright (c) Microsoft. All rights reserved.

"""Workflow fixtures for hosting tests.

Defined in a module that does not use ``from __future__ import annotations``
because the workflow handler validation reflects on real annotation objects
rather than stringified forms.
"""

from typing import Any

from agent_framework import Executor, Workflow, WorkflowBuilder, WorkflowContext, handler


class _UpperExecutor(Executor):
    """Represents the _UpperExecutor type and related behavior.
    """
    @handler
    async def handle(self, text: str, ctx: WorkflowContext[Any, str]) -> None:
        """Implements handle.
        
        Args:
            self: Description of self.
            text: Description of text.
            ctx: Description of ctx.
        """
        await ctx.yield_output(text.upper())


class _EchoExecutor(Executor):
    """Represents the _EchoExecutor type and related behavior.
    """
    @handler
    async def handle(self, text: str, ctx: WorkflowContext[Any, str]) -> None:
        """Implements handle.
        
        Args:
            self: Description of self.
            text: Description of text.
            ctx: Description of ctx.
        """
        await ctx.yield_output(text)


def build_upper_workflow() -> Workflow:
    """Implements build upper workflow.
    
    Returns:
        Description of the return value.
    """
    return WorkflowBuilder(start_executor=_UpperExecutor(id="upper")).build()


def build_echo_workflow() -> Workflow:
    """Implements build echo workflow.
    
    Returns:
        Description of the return value.
    """
    return WorkflowBuilder(start_executor=_EchoExecutor(id="echo")).build()


def echo_workflow_builder() -> WorkflowBuilder:
    """Return an *unbuilt* echo ``WorkflowBuilder``, for testing builder-shaped targets."""
    return WorkflowBuilder(start_executor=_EchoExecutor(id="echo"))


class _MultiChunkExecutor(Executor):
    """Yields three separate ``output`` events so streaming has something to chew on."""

    @handler
    async def handle(self, text: str, ctx: WorkflowContext[Any, str]) -> None:
        """Implements handle.
        
        Args:
            self: Description of self.
            text: Description of text.
            ctx: Description of ctx.
        """
        for chunk in (f"{text}-1", f"{text}-2", f"{text}-3"):
            await ctx.yield_output(chunk)


def build_multi_chunk_workflow() -> Workflow:
    """Implements build multi chunk workflow.
    
    Returns:
        Description of the return value.
    """
    return WorkflowBuilder(start_executor=_MultiChunkExecutor(id="multi")).build()
