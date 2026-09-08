"""This module defines functionality for packages/lab/namespace/agent_framework/__init__.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

# This makes agent_framework a namespace package
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
