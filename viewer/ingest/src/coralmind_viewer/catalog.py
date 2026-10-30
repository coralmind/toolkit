"""Discover simulation and judge JSONL files under a results directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Batch:
    """One batch directory, or the run directory when JSONL sits at its root."""

    name: str
    directory: Path
    sim_files: tuple[Path, ...]
    judge_files: tuple[Path, ...]
    check_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Run:
    """One results directory: the manifest ``run_dir`` name and its batches."""

    name: str
    directory: Path
    batches: tuple[Batch, ...]


def scan(results_root: Path) -> list[Run]:
    """List run directories under ``results_root`` that contain sim or judge JSONL."""
    if not results_root.is_dir():
        return []
    runs: list[Run] = []
    for child in sorted(p for p in results_root.iterdir() if p.is_dir()):
        batches = _batches(child)
        if batches:
            runs.append(Run(name=child.name, directory=child, batches=tuple(batches)))
    return runs


def _batches(run_dir: Path) -> list[Batch]:
    batches: list[Batch] = []
    root_batch = _batch_from_dir(run_dir, "")
    if root_batch.sim_files or root_batch.judge_files:
        batches.append(root_batch)
    batch_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("batch-"))
    batches.extend(_batch_from_dir(sub, sub.name) for sub in batch_dirs)
    return batches


def _batch_from_dir(directory: Path, name: str) -> Batch:
    sim_files = tuple(sorted(directory.glob("*-sim.jsonl")))
    judge_files = tuple(sorted(directory.glob("*-judge.jsonl")))
    check_files = tuple(sorted(directory.glob("*-check.jsonl")))
    return Batch(
        name=name,
        directory=directory,
        sim_files=sim_files,
        judge_files=judge_files,
        check_files=check_files,
    )
