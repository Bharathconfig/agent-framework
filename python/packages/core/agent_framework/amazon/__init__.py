# Copyright (c) Microsoft. All rights reserved.

"""Amazon Bedrock integration namespace for optional Agent Framework connectors.

This module lazily re-exports objects from:
- ``agent-framework-anthropic``
- ``agent-framework-bedrock``

Supported classes:
- AnthropicBedrockClient
- BedrockChatClient
- BedrockChatOptions
- BedrockEmbeddingClient
- BedrockEmbeddingOptions
- BedrockEmbeddingSettings
- BedrockGuardrailConfig
- BedrockSettings
- RawAnthropicBedrockClient
"""

import importlib
from typing import Any

_IMPORTS: dict[str, tuple[str, str]] = {
    "AnthropicBedrockClient": ("agent_framework_anthropic", "agent-framework-anthropic"),
    "BedrockChatClient": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "BedrockChatOptions": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "BedrockEmbeddingClient": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "BedrockEmbeddingOptions": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "BedrockEmbeddingSettings": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "BedrockGuardrailConfig": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "BedrockSettings": ("agent_framework_bedrock", "agent-framework-bedrock"),
    "RawAnthropicBedrockClient": ("agent_framework_anthropic", "agent-framework-anthropic"),
}


def __getattr__(name: str) -> Any:
    """Implements   getattr  .
    
    Args:
        name: Description of name.
    
    Returns:
        Description of the return value.
    """
    if name in _IMPORTS:
        import_path, package_name = _IMPORTS[name]
        try:
            return getattr(importlib.import_module(import_path), name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"The '{package_name}' package is not installed, please do `pip install {package_name}`"
            ) from exc
    raise AttributeError(f"Module `amazon` has no attribute {name}.")


def __dir__() -> list[str]:
    """Implements   dir  .
    
    Returns:
        Description of the return value.
    """
    return list(_IMPORTS.keys())
