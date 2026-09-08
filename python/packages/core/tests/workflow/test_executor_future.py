"""This module defines functionality for packages/core/tests/workflow/test_executor_future.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_framework import Executor, WorkflowContext, handler


class MyTypeA(BaseModel):
    """Represents the MyTypeA type and related behavior.
    """
    pass


class MyTypeB(BaseModel):
    """Represents the MyTypeB type and related behavior.
    """
    pass


class MyTypeC(BaseModel):
    """Represents the MyTypeC type and related behavior.
    """
    pass


class TestExecutorFutureAnnotations:
    """Test suite for Executor with from __future__ import annotations."""

    def test_handler_decorator_future_annotations(self):
        """Test @handler decorator works with stringified annotations (issue #3898)."""

        class MyExecutor(Executor):
            """Represents the MyExecutor type and related behavior.
            """
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:
                """Implements example.
                
                Args:
                    self: Description of self.
                    input: Description of input.
                    ctx: Description of ctx.
                """
                pass

        exec_instance = MyExecutor(id="test")
        assert str in exec_instance._handlers  # pyright: ignore[reportPrivateUsage]
        spec = exec_instance._handler_specs[0]  # pyright: ignore[reportPrivateUsage]
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]
        assert spec["workflow_output_types"] == [MyTypeB]

    def test_handler_decorator_future_annotations_single_type_arg(self):
        """Test @handler with single type argument and future annotations."""

        class MyExecutor(Executor):
            """Represents the MyExecutor type and related behavior.
            """
            @handler
            async def example(self, input: int, ctx: WorkflowContext[MyTypeA]) -> None:
                """Implements example.
                
                Args:
                    self: Description of self.
                    input: Description of input.
                    ctx: Description of ctx.
                """
                pass

        exec_instance = MyExecutor(id="test")
        assert int in exec_instance._handlers  # pyright: ignore[reportPrivateUsage]
        spec = exec_instance._handler_specs[0]  # pyright: ignore[reportPrivateUsage]
        assert spec["message_type"] is int
        assert spec["output_types"] == [MyTypeA]

    def test_handler_decorator_future_annotations_complex(self):
        """Test @handler with complex type annotations and future annotations."""

        class MyExecutor(Executor):
            """Represents the MyExecutor type and related behavior.
            """
            @handler
            async def example(self, data: dict[str, Any], ctx: WorkflowContext[list[str]]) -> None:
                """Implements example.
                
                Args:
                    self: Description of self.
                    data: Description of data.
                    ctx: Description of ctx.
                """
                pass

        exec_instance = MyExecutor(id="test")
        spec = exec_instance._handler_specs[0]  # pyright: ignore[reportPrivateUsage]
        assert spec["message_type"] == dict[str, Any]
        assert spec["output_types"] == [list[str]]

    def test_handler_decorator_future_annotations_bare_context(self):
        """Test @handler with bare WorkflowContext and future annotations."""

        class MyExecutor(Executor):
            """Represents the MyExecutor type and related behavior.
            """
            @handler
            async def example(self, input: str, ctx: WorkflowContext) -> None:
                """Implements example.
                
                Args:
                    self: Description of self.
                    input: Description of input.
                    ctx: Description of ctx.
                """
                pass

        exec_instance = MyExecutor(id="test")
        assert str in exec_instance._handlers  # pyright: ignore[reportPrivateUsage]
        spec = exec_instance._handler_specs[0]  # pyright: ignore[reportPrivateUsage]
        assert spec["output_types"] == []
        assert spec["workflow_output_types"] == []

    def test_handler_decorator_future_annotations_explicit_types(self):
        """Test @handler with explicit type parameters under future annotations."""

        class MyExecutor(Executor):
            """Represents the MyExecutor type and related behavior.
            """
            @handler(input=str, output=MyTypeA)
            async def example(self, input, ctx) -> None:  # type: ignore[no-untyped-def]
                """Implements example.
                
                Args:
                    self: Description of self.
                    input: Description of input.
                    ctx: Description of ctx.
                """
                pass

        exec_instance = MyExecutor(id="test")
        assert str in exec_instance._handlers  # pyright: ignore[reportPrivateUsage]
        spec = exec_instance._handler_specs[0]  # pyright: ignore[reportPrivateUsage]
        assert spec["message_type"] is str
        assert spec["output_types"] == [MyTypeA]

    def test_handler_decorator_future_annotations_union_context(self):
        """Test @handler with union type context annotations and future annotations."""

        class MyExecutor(Executor):
            """Represents the MyExecutor type and related behavior.
            """
            @handler
            async def example(self, input: str, ctx: WorkflowContext[MyTypeA | MyTypeB, MyTypeC]) -> None:
                """Implements example.
                
                Args:
                    self: Description of self.
                    input: Description of input.
                    ctx: Description of ctx.
                """
                pass

        exec_instance = MyExecutor(id="test")
        assert str in exec_instance._handlers  # pyright: ignore[reportPrivateUsage]
        spec = exec_instance._handler_specs[0]  # pyright: ignore[reportPrivateUsage]
        assert spec["output_types"] == [MyTypeA, MyTypeB]
        assert spec["workflow_output_types"] == [MyTypeC]

    def test_handler_unresolvable_annotation_raises(self):
        """Test that an unresolvable forward-reference annotation raises ValueError.

        When get_type_hints fails (e.g. NameError for NonExistentType), the code falls back
        to raw string annotations. The ctx parameter's raw string annotation is then not
        recognised as a valid WorkflowContext type, so a ValueError is still raised.
        """
        with pytest.raises(ValueError):

            class Bad(Executor):  # pyright: ignore[reportUnusedClass]
                """Represents the Bad type and related behavior.
                """
                @handler  # pyright: ignore[reportUnknownArgumentType]
                async def example(self, input: NonExistentType, ctx: WorkflowContext[MyTypeA, MyTypeB]) -> None:  # type: ignore[name-defined]  # ty: ignore[unresolved-reference]  # noqa: F821
                    """Implements example.
                    
                    Args:
                        self: Description of self.
                        input: Description of input.
                        ctx: Description of ctx.
                    """
                    pass
