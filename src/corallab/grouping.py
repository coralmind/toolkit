"""Split a run into consecutive groups of a fixed size.

Simulation uses this to cap how many test cases are in flight. Each batch runs
to completion and is written under its own directory before the next batch
starts. Judging can call the same helpers when it grows a batch cap; it does
not yet.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path


def iter_groups[T](items: Sequence[T], batch_size: int) -> Iterator[tuple[int, list[T]]]:
    """Yield ``(1-based index, group)`` slices of ``batch_size``.

    The last group may be shorter. An empty sequence yields nothing.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for index, start in enumerate(range(0, len(items), batch_size), start=1):
        yield index, list(items[start : start + batch_size])


def batch_directory(parent: Path, index: int) -> Path:
    """Directory for batch ``index`` (1-based) under ``parent``."""
    return parent / f"batch-{index:03d}"
