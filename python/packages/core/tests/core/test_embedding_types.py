"""This module defines functionality for packages/core/tests/core/test_embedding_types.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from datetime import datetime

from agent_framework import Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings

# --- Embedding tests ---


def test_embedding_basic_construction() -> None:
    """Validates behavior for embedding basic construction.
    """
    embedding = Embedding(vector=[0.1, 0.2, 0.3])
    assert embedding.vector == [0.1, 0.2, 0.3]
    assert embedding.model is None
    assert embedding.created_at is None
    assert embedding.additional_properties == {}


def test_embedding_construction_with_metadata() -> None:
    """Validates behavior for embedding construction with metadata.
    """
    now = datetime.now()
    embedding = Embedding(
        vector=[0.1, 0.2],
        model="text-embedding-3-small",
        created_at=now,
        additional_properties={"key": "value"},
    )
    assert embedding.model == "text-embedding-3-small"
    assert embedding.created_at == now
    assert embedding.additional_properties == {"key": "value"}


def test_embedding_dimensions_computed_from_list() -> None:
    """Validates behavior for embedding dimensions computed from list.
    """
    embedding = Embedding(vector=[0.1, 0.2, 0.3])
    assert embedding.dimensions == 3


def test_embedding_dimensions_computed_from_tuple() -> None:
    """Validates behavior for embedding dimensions computed from tuple.
    """
    embedding = Embedding(vector=(0.1, 0.2, 0.3, 0.4))
    assert embedding.dimensions == 4


def test_embedding_dimensions_computed_from_bytes() -> None:
    """Validates behavior for embedding dimensions computed from bytes.
    """
    embedding = Embedding(vector=b"\x00\x01\x02")
    assert embedding.dimensions == 3


def test_embedding_dimensions_explicit_overrides_computed() -> None:
    """Validates behavior for embedding dimensions explicit overrides computed.
    """
    embedding = Embedding(vector=[0.1, 0.2, 0.3], dimensions=1536)
    assert embedding.dimensions == 1536


def test_embedding_dimensions_none_for_unknown_type() -> None:
    """Validates behavior for embedding dimensions none for unknown type.
    """
    embedding = Embedding(vector="not a list")  # type: ignore[arg-type]
    assert embedding.dimensions is None


def test_embedding_dimensions_explicit_with_unknown_type() -> None:
    """Validates behavior for embedding dimensions explicit with unknown type.
    """
    embedding = Embedding(vector="not a list", dimensions=100)  # type: ignore[arg-type]
    assert embedding.dimensions == 100


def test_embedding_empty_vector() -> None:
    """Validates behavior for embedding empty vector.
    """
    embedding = Embedding(vector=[])  # type: ignore[var-annotated]
    assert embedding.dimensions == 0


def test_embedding_int_vector() -> None:
    """Validates behavior for embedding int vector.
    """
    embedding = Embedding(vector=[1, 2, 3])
    assert embedding.vector == [1, 2, 3]
    assert embedding.dimensions == 3


# --- GeneratedEmbeddings tests ---


def test_generated_basic_construction() -> None:
    """Validates behavior for generated basic construction.
    """
    embeddings = GeneratedEmbeddings()
    assert len(embeddings) == 0
    assert embeddings.options is None
    assert embeddings.usage is None
    assert embeddings.additional_properties == {}


def test_generated_construction_with_embeddings() -> None:
    """Validates behavior for generated construction with embeddings.
    """
    items = [Embedding(vector=[0.1, 0.2]), Embedding(vector=[0.3, 0.4])]
    embeddings = GeneratedEmbeddings(items)
    assert len(embeddings) == 2
    assert embeddings[0].vector == [0.1, 0.2]
    assert embeddings[1].vector == [0.3, 0.4]


def test_generated_construction_with_usage() -> None:
    """Validates behavior for generated construction with usage.
    """
    usage = {"prompt_tokens": 10, "total_tokens": 10}
    embeddings = GeneratedEmbeddings(
        [
            Embedding(
                vector=[0.1],
                model="test-model",
            )
        ],
        usage=usage,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )
    assert embeddings.usage == usage
    assert embeddings.usage["prompt_tokens"] == 10  # type: ignore[index, typeddict-item]  # pyrefly: ignore[unsupported-operation]  # ty: ignore[not-subscriptable]


def test_generated_construction_with_additional_properties() -> None:
    """Validates behavior for generated construction with additional properties.
    """
    embeddings = GeneratedEmbeddings(
        additional_properties={"model": "test"},
    )
    assert embeddings.additional_properties == {"model": "test"}


def test_generated_construction_with_options() -> None:
    """Validates behavior for generated construction with options.
    """
    opts: EmbeddingGenerationOptions = {"model": "text-embedding-3-small", "dimensions": 256}
    embeddings = GeneratedEmbeddings(
        [Embedding(vector=[0.1])],
        options=opts,
    )
    assert embeddings.options is not None
    assert embeddings.options["model"] == "text-embedding-3-small"
    assert embeddings.options["dimensions"] == 256


def test_generated_list_behavior_iteration() -> None:
    """Validates behavior for generated list behavior iteration.
    """
    items = [Embedding(vector=[float(i)]) for i in range(5)]
    embeddings = GeneratedEmbeddings(items)
    vectors = [e.vector for e in embeddings]
    assert vectors == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_generated_list_behavior_indexing() -> None:
    """Validates behavior for generated list behavior indexing.
    """
    items = [Embedding(vector=[0.1]), Embedding(vector=[0.2])]
    embeddings = GeneratedEmbeddings(items)
    assert embeddings[0].vector == [0.1]
    assert embeddings[-1].vector == [0.2]


def test_generated_list_behavior_slicing() -> None:
    """Validates behavior for generated list behavior slicing.
    """
    items = [Embedding(vector=[float(i)]) for i in range(5)]
    embeddings = GeneratedEmbeddings(items)
    sliced = embeddings[1:3]
    assert len(sliced) == 2


def test_generated_list_behavior_append() -> None:
    """Validates behavior for generated list behavior append.
    """
    embeddings = GeneratedEmbeddings()
    embeddings.append(Embedding(vector=[0.1]))
    assert len(embeddings) == 1


def test_generated_none_embeddings_creates_empty_list() -> None:
    """Validates behavior for generated none embeddings creates empty list.
    """
    embeddings = GeneratedEmbeddings(None)
    assert len(embeddings) == 0


# --- EmbeddingGenerationOptions tests ---


def test_options_empty() -> None:
    """Validates behavior for options empty.
    """
    options: EmbeddingGenerationOptions = {}
    assert "model" not in options


def test_options_with_model() -> None:
    """Validates behavior for options with model.
    """
    options: EmbeddingGenerationOptions = {"model": "text-embedding-3-small"}
    assert options["model"] == "text-embedding-3-small"


def test_options_with_dimensions() -> None:
    """Validates behavior for options with dimensions.
    """
    options: EmbeddingGenerationOptions = {"dimensions": 1536}
    assert options["dimensions"] == 1536


def test_options_with_all_fields() -> None:
    """Validates behavior for options with all fields.
    """
    options: EmbeddingGenerationOptions = {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }
    assert options["model"] == "text-embedding-3-small"
    assert options["dimensions"] == 1536
