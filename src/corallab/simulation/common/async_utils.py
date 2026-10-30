"""Async helpers for cache-aware LLM call fan-outs.

Two patterns for firing many calls that share prompt prefixes:

- ``gather_with_cache_warmup``: two-wave firing. Wave 1 sends one call per
  unique cache_key in parallel (warms the cache). Wave 2 sends all remaining
  calls in parallel — they hit the now-warm cache. Use for moderate bursts
  (≤3 same-key calls at once).

- ``gather_per_key_serial``: calls sharing a cache_key run sequentially (first
  primes the cache, subsequent calls hit it), while calls with different keys
  run in parallel. Use when bursts of ≥4 same-key calls would exceed OpenAI's
  ~15 RPM per-key routing threshold, or when prompts are large enough that
  each cache miss is expensive.

Neither helper is wired into any judge yet; they exist here for the
grouped-incremental judge and any future judge that fans out parallel calls
with shared prefixes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_with_cache_warmup(
    specs: list[tuple],
    build_coro: Callable[[tuple], Awaitable[Any]],
    cache_key_index: int = -1,
) -> list[Any]:
    """Fire calls in two waves so cache prefix is populated before parallel bursts.

    Problem: when many calls share a prompt_cache_key and fire simultaneously via
    asyncio.gather(), none has completed to populate the cache — they all miss.

    Fix: identify the first call per unique cache_key, fire those as a warm-up
    wave and await completion. Then fire remaining calls in parallel — they hit
    the warm cache.

    **Exception contract.** ``build_coro`` MUST NOT raise. If a coroutine raises,
    ``asyncio.gather`` cancels all siblings in that wave and propagates the
    exception up — losing every in-flight result and bypassing any partial-
    failure threshold the caller defines. Wrap per-call errors in ``build_coro``
    and return a sentinel (e.g. a ``JudgeRawOutput(status="judge_error")``) so
    this helper never sees exceptions.

    Args:
        specs: List of spec tuples. Each must include a cache_key at
            ``cache_key_index`` (default: last element).
        build_coro: Function mapping a spec to a coroutine. Must not raise.
        cache_key_index: Position of cache_key within each spec tuple.

    Returns:
        Results in original spec order.
    """
    seen_keys: set[str] = set()
    warmup_idx: list[int] = []
    remaining_idx: list[int] = []

    for i, spec in enumerate(specs):
        cache_key = spec[cache_key_index]
        if cache_key in seen_keys:
            remaining_idx.append(i)
        else:
            seen_keys.add(cache_key)
            warmup_idx.append(i)

    logger.info(
        "gather_with_cache_warmup: %d warm-up calls (unique keys), %d parallel calls after",
        len(warmup_idx),
        len(remaining_idx),
    )

    warmup_results = await asyncio.gather(*[build_coro(specs[i]) for i in warmup_idx])

    remaining_results: list[Any] = []
    if remaining_idx:
        remaining_results = list(
            await asyncio.gather(*[build_coro(specs[i]) for i in remaining_idx])
        )

    results: list[Any] = [None] * len(specs)
    for i, r in zip(warmup_idx, warmup_results, strict=True):
        results[i] = r
    for i, r in zip(remaining_idx, remaining_results, strict=True):
        results[i] = r
    return results


async def gather_per_key_serial(
    specs: list[tuple],
    build_coro: Callable[[tuple], Awaitable[Any]],
    cache_key_index: int = -1,
) -> list[Any]:
    """Fire calls with per-cache_key serialization, parallel across keys.

    Calls sharing a cache_key run sequentially (first call primes the cache,
    subsequent calls hit it). Calls with different keys run in parallel.

    Use when bursts of same-key calls exceed OpenAI's ~15 RPM per-key routing
    threshold — firing 4+ same-key calls simultaneously causes some to overflow
    to cold shards.

    **Exception contract.** ``build_coro`` MUST NOT raise. A raise on call N
    of a key aborts calls N+1..end for that key AND cancels every other key's
    in-flight work via ``asyncio.gather`` — results already collected in this
    function are lost. Wrap per-call errors in ``build_coro`` and return a
    sentinel (e.g. a ``JudgeRawOutput(status="judge_error")``) so this helper
    never sees exceptions, and the caller's partial-failure threshold stays
    the single failure path.

    Args:
        specs: List of spec tuples. Each must include a cache_key at
            ``cache_key_index`` (default: last element).
        build_coro: Function mapping a spec to a coroutine. Must not raise.
        cache_key_index: Position of cache_key within each spec tuple.

    Returns:
        Results in original spec order.
    """
    by_key: dict[str, list[tuple[int, tuple]]] = {}
    for i, spec in enumerate(specs):
        key = spec[cache_key_index]
        by_key.setdefault(key, []).append((i, spec))

    logger.info(
        "gather_per_key_serial: %d unique keys, %d total calls"
        " (sequential within key, parallel across keys)",
        len(by_key),
        len(specs),
    )

    async def _process_key(key_specs: list[tuple[int, tuple]]) -> list[tuple[int, Any]]:
        out = []
        for orig_idx, spec in key_specs:
            r = await build_coro(spec)
            out.append((orig_idx, r))
        return out

    key_results = await asyncio.gather(*[_process_key(v) for v in by_key.values()])

    results: list[Any] = [None] * len(specs)
    for key_batch in key_results:
        for orig_idx, r in key_batch:
            results[orig_idx] = r
    return results
