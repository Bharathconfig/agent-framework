"""
common
======

Shared code used by every example:

* :mod:`common.config` - loads ``.env`` and builds the Azure OpenAI chat client.
* :mod:`common.utils`  - tiny pretty-printing helpers for console output.

Typical usage inside an example::

    from common import get_chat_client, print_header

    client = get_chat_client()
"""

from common.config import get_chat_client, load_settings
from common.utils import print_agent, print_header, print_section

__all__ = ["get_chat_client", "load_settings", "print_agent", "print_header", "print_section"]
