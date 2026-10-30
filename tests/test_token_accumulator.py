"""Tests for TokenAccumulator — totals, per-(role, model) breakdown, and summary rendering.

The accumulator tallies real token usage across a run. It is set on a ContextVar
before asyncio.run() and read after completion. These tests exercise the pure
accounting logic (record/breakdown/totals) and that summary_lines() renders
plain text for both populated and empty accumulators. Console presentation lives
in the CLI layer (cli._helpers.print_token_summary), tested separately.
"""

from __future__ import annotations

from corallab.llm_client import (
    TokenAccumulator,
    get_token_accumulator,
    reset_token_accumulator,
    set_token_accumulator,
)


def test_record_aggregates_totals_and_per_key():
    acc = TokenAccumulator()
    acc.record(role="judge", model="m1", input_tokens=100, output_tokens=10, cache_tokens=5)
    acc.record(role="judge", model="m1", input_tokens=200, output_tokens=20, cache_tokens=15)
    acc.record(role="sim", model="m2", input_tokens=50, output_tokens=5, cache_tokens=0)

    # Totals sum across every record.
    assert acc.total_input == 350
    assert acc.total_output == 35
    assert acc.total_cache == 20

    # Same (role, model) key accumulates; distinct key is separate.
    bd = acc.breakdown
    assert bd[("judge", "m1")] == (300, 30, 20)
    assert bd[("sim", "m2")] == (50, 5, 0)


def test_breakdown_sorted_by_role_then_model():
    acc = TokenAccumulator()
    # Insert out of sorted order to prove the property sorts, not insertion-orders.
    acc.record(role="sim", model="zeta", input_tokens=1, output_tokens=1, cache_tokens=1)
    acc.record(role="judge", model="beta", input_tokens=1, output_tokens=1, cache_tokens=1)
    acc.record(role="judge", model="alpha", input_tokens=1, output_tokens=1, cache_tokens=1)

    assert list(acc.breakdown.keys()) == [
        ("judge", "alpha"),
        ("judge", "beta"),
        ("sim", "zeta"),
    ]


def test_empty_accumulator_has_zero_totals_and_empty_breakdown():
    acc = TokenAccumulator()
    assert acc.total_input == 0
    assert acc.total_output == 0
    assert acc.total_cache == 0
    assert acc.breakdown == {}


def test_summary_lines_render_text():
    acc = TokenAccumulator()
    acc.record(role="judge", model="claude-x", input_tokens=1234, output_tokens=56, cache_tokens=7)
    text = "\n".join(acc.summary_lines("judge"))

    assert "Token usage — judge" in text
    # Thousands separators are applied to the figures.
    assert "1,234" in text
    assert "TOTAL" in text


def test_summary_lines_on_empty_accumulator_does_not_raise():
    # Empty path uses max(..., default=...) for column widths — must not crash.
    text = "\n".join(TokenAccumulator().summary_lines("sim"))
    assert "TOTAL" in text


def test_contextvar_set_get_reset_roundtrip():
    assert get_token_accumulator() is None
    acc = TokenAccumulator()
    token = set_token_accumulator(acc)
    try:
        assert get_token_accumulator() is acc
    finally:
        reset_token_accumulator(token)
    assert get_token_accumulator() is None
