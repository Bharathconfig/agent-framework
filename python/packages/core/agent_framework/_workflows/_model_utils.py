"""This module defines functionality for packages/core/agent_framework/_workflows/_model_utils.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

import copy
import sys
from typing import Any, TypeVar, cast

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self  # pragma: no cover

ModelT = TypeVar("ModelT", bound="DictConvertible")


class DictConvertible:
    """Mixin providing conversion helpers for plain Python models."""

    def to_dict(self) -> dict[str, Any]:
        """Implements to dict.
        
        Args:
            self: Description of self.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls: type[ModelT], data: dict[str, Any]) -> ModelT:
        """Implements from dict.
        
        Args:
            cls: Description of cls.
            data: Description of data.
        
        Returns:
            Description of the return value.
        """
        return cls(**data)

    def clone(self, *, deep: bool = True) -> Self:
        """Implements clone.
        
        Args:
            self: Description of self.
            deep: Description of deep.
        
        Returns:
            Description of the return value.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def to_json(self) -> str:
        """Implements to json.
        
        Args:
            self: Description of self.
        
        Returns:
            Description of the return value.
        """
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls: type[ModelT], raw: str) -> ModelT:
        """Implements from json.
        
        Args:
            cls: Description of cls.
            raw: Description of raw.
        
        Returns:
            Description of the return value.
        """
        import json

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON payload must decode to a mapping")
        return cls.from_dict(cast(dict[str, Any], data))


def encode_value(value: Any) -> Any:
    """Recursively encode values for JSON-friendly serialization."""
    if isinstance(value, DictConvertible):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: encode_value(v) for k, v in value.items()}  # type: ignore[misc]
    if isinstance(value, (list, tuple, set)):
        return [encode_value(v) for v in value]  # type: ignore[misc]
    return value
