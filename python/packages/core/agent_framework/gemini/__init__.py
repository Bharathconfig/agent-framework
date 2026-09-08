# Copyright (c) Microsoft. All rights reserved.

"""Google Gemini namespace for optional Agent Framework connectors.

This module lazily re-exports objects from ``agent-framework-gemini``.
"""

import importlib
from typing import Any

_IMPORTS: dict[str, tuple[str, str]] = {
    "GeminiChatClient": ("agent_framework_gemini", "agent-framework-gemini"),
    "GeminiChatOptions": ("agent_framework_gemini", "agent-framework-gemini"),
    "GeminiSettings": ("agent_framework_gemini", "agent-framework-gemini"),
    "GoogleGeminiSettings": ("agent_framework_gemini", "agent-framework-gemini"),
    "RawGeminiChatClient": ("agent_framework_gemini", "agent-framework-gemini"),
    "ThinkingConfig": ("agent_framework_gemini", "agent-framework-gemini"),
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
                f"The package {package_name} is required to use `{name}`. "
                f"Please use `pip install {package_name}`, or update your requirements.txt or pyproject.toml file."
            ) from exc
    raise AttributeError(f"Module `gemini` has no attribute {name}.")


def __dir__() -> list[str]:
    """Implements   dir  .
    
    Returns:
        Description of the return value.
    """
    return list(_IMPORTS.keys())
