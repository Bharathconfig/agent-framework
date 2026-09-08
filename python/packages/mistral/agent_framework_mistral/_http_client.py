"""This module defines functionality for packages/mistral/agent_framework_mistral/_http_client.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from typing import Any

import httpx


class AsyncClientUsingConfiguredTimeout:
    """Let an injected HTTPX client retain its configured per-phase timeouts."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Initializes a new instance.
        
        Args:
            self: Description of self.
            client: Description of client.
        """
        self.client = client

    def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
        """Implements build request.
        
        Args:
            self: Description of self.
            *args: Description of args.
            **kwargs: Description of kwargs.
        
        Returns:
            Description of the return value.
        """
        kwargs["timeout"] = httpx.USE_CLIENT_DEFAULT
        return self.client.build_request(*args, **kwargs)

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Implements send.
        
        Args:
            self: Description of self.
            request: Description of request.
            **kwargs: Description of kwargs.
        
        Returns:
            Description of the return value.
        """
        return await self.client.send(request, **kwargs)

    async def aclose(self) -> None:
        # The caller owns the injected client.
        """Implements aclose.
        
        Args:
            self: Description of self.
        """
        return
