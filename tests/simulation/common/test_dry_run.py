"""Tests for eval/common/dry_run.py — no LLM calls, no Burr required."""

from __future__ import annotations

import re

from corallab.simulation.common.dry_run import DryRunCollector

_MODEL = "azure/gpt-4o"


def _record(
    collector: DryRunCollector,
    *,
    cache_key: str | None,
    prompt_tokens: int,
    trace_role: str = "judge",
) -> None:
    collector.record(
        cache_key=cache_key,
        trace_role=trace_role,
        prompt_tokens=prompt_tokens,
        provider_model=_MODEL,
    )


# ---------------------------------------------------------------------------
# record() accumulation
# ---------------------------------------------------------------------------


class TestRecord:
    def test_single_record_stored(self):
        c = DryRunCollector()
        _record(c, cache_key="k1", prompt_tokens=100)
        assert len(c._records) == 1
        assert c._records[0].cache_key == "k1"
        assert c._records[0].prompt_tokens == 100

    def test_multiple_records_accumulated(self):
        c = DryRunCollector()
        for i in range(5):
            _record(c, cache_key="k1", prompt_tokens=100 + i)
        assert len(c._records) == 5

    def test_none_cache_key_accepted(self):
        c = DryRunCollector()
        _record(c, cache_key=None, prompt_tokens=50)
        assert c._records[0].cache_key is None

    def test_empty_collector_renders_no_calls(self):
        c = DryRunCollector()
        out = c.render_summary()
        assert "no LLM calls recorded" in out


# ---------------------------------------------------------------------------
# Hit rate formula: hits[k] = N - 1 (first call always cold)
# ---------------------------------------------------------------------------


class TestHitRateFormula:
    def test_single_call_no_cache_hits(self):
        c = DryRunCollector()
        _record(c, cache_key="k1", prompt_tokens=1000)
        out = c.render_summary()
        assert "0.0%" in out

    def test_two_calls_fifty_percent_hit_rate(self):
        c = DryRunCollector()
        for _ in range(2):
            _record(c, cache_key="k1", prompt_tokens=1000)
        out = c.render_summary()
        assert "50.0%" in out

    def test_ten_calls_ninety_percent_hit_rate(self):
        c = DryRunCollector()
        for _ in range(10):
            _record(c, cache_key="k1", prompt_tokens=1000)
        out = c.render_summary()
        assert "90.0%" in out


# ---------------------------------------------------------------------------
# stable_tokens = first record's prompt_tokens
# ---------------------------------------------------------------------------


class TestStableTokens:
    def test_stable_tok_uses_first_call(self):
        c = DryRunCollector()
        _record(c, cache_key="k1", prompt_tokens=5000)
        _record(c, cache_key="k1", prompt_tokens=5100)
        _record(c, cache_key="k1", prompt_tokens=5200)
        out = c.render_summary()
        assert "5,000" in out


# ---------------------------------------------------------------------------
# Multiple cache keys
# ---------------------------------------------------------------------------


class TestMultipleCacheKeys:
    def test_separate_rows_per_key(self):
        c = DryRunCollector()
        for _ in range(3):
            _record(c, cache_key="group_a", prompt_tokens=1000)
        for _ in range(5):
            _record(c, cache_key="group_b", prompt_tokens=2000)
        out = c.render_summary()
        assert "group_a" in out
        assert "group_b" in out

    def test_total_row_sums_all_calls(self):
        c = DryRunCollector()
        for _ in range(3):
            _record(c, cache_key="group_a", prompt_tokens=1000)
        for _ in range(5):
            _record(c, cache_key="group_b", prompt_tokens=1000)
        out = c.render_summary()
        assert "TOTAL" in out
        assert re.search(r"TOTAL\s+8\b", out)
