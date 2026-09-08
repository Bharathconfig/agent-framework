"""This module defines functionality for packages/core/tests/core/test_embedding_client.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agent_framework import (
    BaseEmbeddingClient,
    Embedding,
    EmbeddingGenerationOptions,
    GeneratedEmbeddings,
    SupportsGetEmbeddings,
)


class MockEmbeddingClient(BaseEmbeddingClient):
    """A simple mock embedding client for testing."""

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: EmbeddingGenerationOptions | None = None,
    ) -> GeneratedEmbeddings[list[float]]:
        """Implements get embeddings.
        
        Args:
            self: Description of self.
            values: Description of values.
            options: Description of options.
        
        Returns:
            Description of the return value.
        """
        return GeneratedEmbeddings(
            [Embedding(vector=[0.1, 0.2, 0.3], model="mock-model") for _ in values],
            usage={"prompt_tokens": len(values), "total_tokens": len(values)},  # type: ignore[arg-type]
        )


# --- BaseEmbeddingClient tests ---


async def test_base_get_embeddings() -> None:
    """Validates behavior for base get embeddings.
    """
    client = MockEmbeddingClient()
    result = await client.get_embeddings(["hello", "world"])
    assert len(result) == 2
    assert result[0].vector == [0.1, 0.2, 0.3]
    assert result[0].model == "mock-model"


async def test_base_get_embeddings_with_options() -> None:
    """Validates behavior for base get embeddings with options.
    """
    client = MockEmbeddingClient()
    options: EmbeddingGenerationOptions = {"model": "test", "dimensions": 3}
    result = await client.get_embeddings(["hello"], options=options)
    assert len(result) == 1


async def test_base_get_embeddings_usage() -> None:
    """Validates behavior for base get embeddings usage.
    """
    client = MockEmbeddingClient()
    result = await client.get_embeddings(["a", "b", "c"])
    assert result.usage is not None
    assert result.usage["prompt_tokens"] == 3  # type: ignore[typeddict-item]


def test_base_additional_properties_default() -> None:
    """Validates behavior for base additional properties default.
    """
    client = MockEmbeddingClient()
    assert client.additional_properties == {}


def test_base_additional_properties_custom() -> None:
    """Validates behavior for base additional properties custom.
    """
    client = MockEmbeddingClient(additional_properties={"key": "value"})
    assert client.additional_properties == {"key": "value"}


def test_base_embedding_client_rejects_unknown_kwargs() -> None:
    """Validates behavior for base embedding client rejects unknown kwargs.
    """
    with pytest.raises(TypeError):
        MockEmbeddingClient(legacy_key="value")  # type: ignore[call-arg]  # ty: ignore[unknown-argument]


# --- SupportsGetEmbeddings protocol tests ---


def test_mock_client_satisfies_protocol() -> None:
    """Validates behavior for mock client satisfies protocol.
    """
    client = MockEmbeddingClient()
    assert isinstance(client, SupportsGetEmbeddings)


def test_plain_class_satisfies_protocol() -> None:
    """A plain class with the right signature should satisfy the protocol."""

    class PlainEmbeddingClient:
        """Represents the PlainEmbeddingClient type and related behavior.
        """
        additional_properties: dict = {}

        async def get_embeddings(self, values, *, options=None):
            """Implements get embeddings.
            
            Args:
                self: Description of self.
                values: Description of values.
                options: Description of options.
            
            Returns:
                Description of the return value.
            """
            return GeneratedEmbeddings()

    client = PlainEmbeddingClient()
    assert isinstance(client, SupportsGetEmbeddings)


def test_wrong_class_does_not_satisfy_protocol() -> None:
    """A class without get_embeddings should not satisfy the protocol."""

    class NotAnEmbeddingClient:
        """Represents the NotAnEmbeddingClient type and related behavior.
        """
        additional_properties: dict = {}

        async def generate(self, values):
            """Implements generate.
            
            Args:
                self: Description of self.
                values: Description of values.
            """
            pass

    client = NotAnEmbeddingClient()
    assert not isinstance(client, SupportsGetEmbeddings)
