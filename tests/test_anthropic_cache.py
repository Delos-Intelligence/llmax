"""Tests for Anthropic system/tool cache-breakpoint placement.

Covers the caller-controlled breakpoint feature and guards the historical
default behaviour (single concatenated system block) against regression.
"""

import anthropic

from llmax.external_clients.anthropic import (
    _MAX_NON_MESSAGE_BREAKPOINTS,
    _convert_tools,
    _extract_system,
)

EPHEMERAL = {"type": "ephemeral"}


def _tools_kwargs() -> dict:
    """Two OpenAI-style function tools, as a fresh kwargs dict."""
    return {
        "tools": [
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "parameters": {}}},
        ],
    }


# --- default mode (string system): must stay byte-identical to historical ---


def test_default_single_string_system() -> None:
    """A single string system message → one block with a trailing breakpoint."""
    system, remaining, n = _extract_system([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ])
    assert n == 1
    assert system == [
        {"type": "text", "text": "You are helpful.", "cache_control": EPHEMERAL},
    ]
    assert remaining == [{"role": "user", "content": "hi"}]


def test_default_multiple_string_systems_are_joined() -> None:
    """Several string system messages are concatenated into one block."""
    system, _remaining, n = _extract_system([
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "hi"},
    ])
    assert n == 1
    assert system == [{"type": "text", "text": "A B", "cache_control": EPHEMERAL}]


def test_no_system_returns_not_given() -> None:
    """No system message → NOT_GIVEN and zero breakpoints."""
    system, remaining, n = _extract_system([{"role": "user", "content": "hi"}])
    assert system is anthropic.NOT_GIVEN
    assert n == 0
    assert remaining == [{"role": "user", "content": "hi"}]


# --- manual mode (caller-supplied blocks with cache_control) ---


def test_manual_two_breakpoints_preserved() -> None:
    """List content with two markers is preserved verbatim (two breakpoints)."""
    system, _remaining, n = _extract_system([
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "cohort", "cache_control": EPHEMERAL},
                {"type": "text", "text": "per-user", "cache_control": EPHEMERAL},
            ],
        },
        {"role": "user", "content": "hi"},
    ])
    assert n == _MAX_NON_MESSAGE_BREAKPOINTS
    assert system == [
        {"type": "text", "text": "cohort", "cache_control": EPHEMERAL},
        {"type": "text", "text": "per-user", "cache_control": EPHEMERAL},
    ]


def test_manual_single_breakpoint_preserved() -> None:
    """One marked block + one unmarked block → a single breakpoint."""
    _system, _remaining, n = _extract_system([
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "cohort", "cache_control": EPHEMERAL},
                {"type": "text", "text": "per-user"},
            ],
        },
    ])
    assert n == 1


def test_manual_list_without_markers_gets_trailing_breakpoint() -> None:
    """List content with no markers falls back to one trailing breakpoint."""
    system, _remaining, n = _extract_system([
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "last"},
            ],
        },
    ])
    assert n == 1
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == EPHEMERAL


# --- tool breakpoint budget ---


def test_tools_cache_enabled_by_default() -> None:
    """By default the last tool carries a cache_control breakpoint."""
    out = _convert_tools(_tools_kwargs())
    assert out["tools"][-1]["cache_control"] == EPHEMERAL


def test_tools_cache_dropped_when_disabled() -> None:
    """enable_cache=False leaves the tools array without a breakpoint."""
    out = _convert_tools(_tools_kwargs(), enable_cache=False)
    assert "cache_control" not in out["tools"][-1]


def test_budget_two_system_breakpoints_drops_tool_breakpoint() -> None:
    """Mirrors anthropic_create: two system breakpoints drop the tools one."""
    _system, _remaining, n = _extract_system([
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "cohort", "cache_control": EPHEMERAL},
                {"type": "text", "text": "per-user", "cache_control": EPHEMERAL},
            ],
        },
    ])
    out = _convert_tools(
        _tools_kwargs(),
        enable_cache=(n < _MAX_NON_MESSAGE_BREAKPOINTS),
    )
    # 2 system + 2 messages (max) + 0 tools = within the 4-breakpoint budget.
    assert "cache_control" not in out["tools"][-1]
