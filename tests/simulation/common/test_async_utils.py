"""Tests for eval/common/async_utils.py cache warmup helpers."""

from __future__ import annotations

import asyncio

import pytest

from corallab.simulation.common.async_utils import (
    gather_per_key_serial,
    gather_with_cache_warmup,
)


async def _echo(spec: tuple) -> tuple:
    """Return the spec unchanged (simulates a trivial LLM call)."""
    return spec


async def _ordered_echo(spec: tuple, log: list) -> tuple:
    """Append to log before returning so tests can verify call order."""
    log.append(spec)
    return spec


# ---------------------------------------------------------------------------
# gather_with_cache_warmup
# ---------------------------------------------------------------------------


class TestGatherWithCacheWarmup:
    @pytest.mark.asyncio
    async def test_returns_all_results_in_original_order(self) -> None:
        specs = [("a", "key1"), ("b", "key2"), ("c", "key1"), ("d", "key2")]
        results = await gather_with_cache_warmup(specs, _echo, cache_key_index=-1)
        assert results == specs

    @pytest.mark.asyncio
    async def test_single_spec_returned(self) -> None:
        specs = [("only", "k")]
        results = await gather_with_cache_warmup(specs, _echo, cache_key_index=-1)
        assert results == specs

    @pytest.mark.asyncio
    async def test_all_unique_keys_no_remaining_wave(self) -> None:
        specs = [("x", "k1"), ("y", "k2"), ("z", "k3")]
        results = await gather_with_cache_warmup(specs, _echo, cache_key_index=-1)
        assert results == specs

    @pytest.mark.asyncio
    async def test_warmup_calls_fire_before_remaining(self) -> None:
        log: list[tuple] = []

        async def tracked(spec: tuple) -> tuple:
            log.append(spec)
            await asyncio.sleep(0)
            return spec

        specs = [("a", "k1"), ("b", "k1"), ("c", "k2"), ("d", "k2")]
        results = await gather_with_cache_warmup(specs, tracked, cache_key_index=-1)
        assert results == specs
        # Warmup set is {("a","k1"), ("c","k2")} — both must appear before remaining
        warmup = {("a", "k1"), ("c", "k2")}
        remaining = {("b", "k1"), ("d", "k2")}
        warmup_positions = [i for i, s in enumerate(log) if s in warmup]
        remaining_positions = [i for i, s in enumerate(log) if s in remaining]
        assert max(warmup_positions) < min(remaining_positions)


# ---------------------------------------------------------------------------
# gather_per_key_serial
# ---------------------------------------------------------------------------


class TestGatherPerKeySerial:
    @pytest.mark.asyncio
    async def test_returns_all_results_in_original_order(self) -> None:
        specs = [("a", "k1"), ("b", "k2"), ("c", "k1"), ("d", "k2")]
        results = await gather_per_key_serial(specs, _echo, cache_key_index=-1)
        assert results == specs

    @pytest.mark.asyncio
    async def test_single_spec(self) -> None:
        specs = [("only", "k")]
        results = await gather_per_key_serial(specs, _echo, cache_key_index=-1)
        assert results == specs

    @pytest.mark.asyncio
    async def test_same_key_calls_run_in_order(self) -> None:
        log: list[tuple] = []

        async def tracked(spec: tuple) -> tuple:
            log.append(spec)
            await asyncio.sleep(0)
            return spec

        # All same key — must run strictly in list order
        specs = [("a", "k"), ("b", "k"), ("c", "k")]
        results = await gather_per_key_serial(specs, tracked, cache_key_index=-1)
        assert results == specs
        assert log == specs

    @pytest.mark.asyncio
    async def test_different_keys_run_in_parallel(self) -> None:
        barrier = asyncio.Event()
        started: list[str] = []

        async def slow(spec: tuple) -> tuple:
            started.append(spec[0])
            await asyncio.wait_for(asyncio.shield(barrier.wait()), timeout=1.0)
            return spec

        async def run():
            task = asyncio.create_task(
                gather_per_key_serial([("a", "k1"), ("b", "k2")], slow, cache_key_index=-1)
            )
            # Give both coroutines a chance to start
            for _ in range(10):
                await asyncio.sleep(0)
                if len(started) >= 2:
                    break
            assert len(started) == 2, "both keys should start before either completes"
            barrier.set()
            return await task

        results = await run()
        assert set(results) == {("a", "k1"), ("b", "k2")}
