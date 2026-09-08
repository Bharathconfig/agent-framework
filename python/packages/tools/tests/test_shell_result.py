"""This module defines functionality for packages/tools/tests/test_shell_result.

This file is part of the Microsoft Agent Framework Python codebase.
"""

# Copyright (c) Microsoft. All rights reserved.

from agent_framework_tools.shell import ShellResult


def _make(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    truncated: bool = False,
    timed_out: bool = False,
) -> ShellResult:
    """Implements  make.
    
    Args:
        stdout: Description of stdout.
        stderr: Description of stderr.
        exit_code: Description of exit_code.
        truncated: Description of truncated.
        timed_out: Description of timed_out.
    
    Returns:
        Description of the return value.
    """
    return ShellResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_ms=1,
        truncated=truncated,
        timed_out=timed_out,
    )


def test_format_stdout_only() -> None:
    """Validates behavior for format stdout only.
    """
    text = _make(stdout="hello").format_for_model()
    assert text == "hello\nexit_code: 0"


def test_format_stdout_truncated_appends_marker() -> None:
    """Validates behavior for format stdout truncated appends marker.
    """
    text = _make(stdout="part", truncated=True).format_for_model()
    assert "[output truncated]" in text
    assert text.startswith("part")


def test_format_stderr_only_truncated_marker() -> None:
    """Validates behavior for format stderr only truncated marker.
    """
    text = _make(stderr="boom", truncated=True, exit_code=1).format_for_model()
    assert "[output truncated]" in text
    assert "stderr: boom" in text


def test_format_truncated_with_empty_streams() -> None:
    """Validates behavior for format truncated with empty streams.
    """
    text = _make(truncated=True).format_for_model()
    assert "[output truncated]" in text
    assert "exit_code: 0" in text


def test_format_stderr_prefixed() -> None:
    """Validates behavior for format stderr prefixed.
    """
    text = _make(stderr="boom", exit_code=1).format_for_model()
    assert "stderr: boom" in text
    assert "exit_code: 1" in text


def test_format_timed_out_marker() -> None:
    """Validates behavior for format timed out marker.
    """
    text = _make(timed_out=True, exit_code=124).format_for_model()
    assert "[command timed out]" in text
    assert "exit_code: 124" in text


def test_format_empty_streams_still_reports_exit_code() -> None:
    """Validates behavior for format empty streams still reports exit code.
    """
    text = _make().format_for_model()
    assert text == "exit_code: 0"


def test_format_combines_all_signals_in_order() -> None:
    """Validates behavior for format combines all signals in order.
    """
    text = _make(
        stdout="out",
        stderr="err",
        exit_code=2,
        truncated=True,
        timed_out=True,
    ).format_for_model()
    lines = text.split("\n")
    assert lines[0] == "out"
    assert lines[1] == "stderr: err"
    assert lines[2] == "[output truncated]"
    assert lines[3] == "[command timed out]"
    assert lines[4] == "exit_code: 2"
