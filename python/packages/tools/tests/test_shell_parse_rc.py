# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the persistent-shell sentinel exit-code parser.

A bug here would silently return -1 for every persistent-mode command's
exit code, masking real failures, so the edge cases are exercised
explicitly even though `_parse_rc` is a private helper.
"""

from __future__ import annotations

from agent_framework_tools.shell._session import _parse_rc


def test_parse_rc_zero() -> None:
    """Validates behavior for parse rc zero.
    """
    assert _parse_rc(b"_0\n") == 0


def test_parse_rc_positive() -> None:
    """Validates behavior for parse rc positive.
    """
    assert _parse_rc(b"_127\n") == 127


def test_parse_rc_negative() -> None:
    """Validates behavior for parse rc negative.
    """
    assert _parse_rc(b"_-1\n") == -1


def test_parse_rc_crlf() -> None:
    """Validates behavior for parse rc crlf.
    """
    assert _parse_rc(b"_42\r\n") == 42


def test_parse_rc_no_trailing_newline() -> None:
    """Validates behavior for parse rc no trailing newline.
    """
    assert _parse_rc(b"_5") == 5


def test_parse_rc_missing_underscore_returns_minus_one() -> None:
    """Validates behavior for parse rc missing underscore returns minus one.
    """
    assert _parse_rc(b"42\n") == -1


def test_parse_rc_empty_returns_minus_one() -> None:
    """Validates behavior for parse rc empty returns minus one.
    """
    assert _parse_rc(b"") == -1


def test_parse_rc_only_underscore_returns_minus_one() -> None:
    """Validates behavior for parse rc only underscore returns minus one.
    """
    assert _parse_rc(b"_\n") == -1


def test_parse_rc_non_digit_returns_minus_one() -> None:
    """Validates behavior for parse rc non digit returns minus one.
    """
    assert _parse_rc(b"_abc\n") == -1


def test_parse_rc_stops_at_first_non_digit() -> None:
    # Trailing garbage after the digits should not corrupt the parse.
    """Validates behavior for parse rc stops at first non digit.
    """
    assert _parse_rc(b"_7 extra junk\n") == 7


def test_parse_rc_partial_digits_then_garbage() -> None:
    """Validates behavior for parse rc partial digits then garbage.
    """
    assert _parse_rc(b"_12x34\n") == 12
