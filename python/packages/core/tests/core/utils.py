"""This module defines functionality for packages/core/tests/core/utils.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.


from copy import deepcopy
from unittest.mock import MagicMock


class CopyingMock(MagicMock):
    """Represents the CopyingMock type and related behavior.
    """
    def __call__(self, *args, **kwargs):
        """Implements   call  .
        
        Args:
            self: Description of self.
            *args: Description of args.
            **kwargs: Description of kwargs.
        
        Returns:
            Description of the return value.
        """
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        return super().__call__(*args, **kwargs)
