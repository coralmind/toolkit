"""The published samples hold only allowed files and stay readable by the current ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from corallab.paths import default_repo_root
from coralmind_viewer import export as E
from coralmind_viewer import facts as F

SAMPLES = default_repo_root() / "data-repository" / "sample_results"


@pytest.fixture
def samples() -> Path:
    if not SAMPLES.is_dir():
        pytest.skip("no sample runs are published yet")
    return SAMPLES


def test_samples_hold_only_the_files_pages_read(samples: Path) -> None:
    from publish_sample import allowed

    extra = [
        p.relative_to(samples).as_posix()
        for p in samples.rglob("*")
        if p.is_file()
        and not allowed(p.relative_to(samples / p.relative_to(samples).parts[0]).as_posix())
    ]
    assert extra == []


def test_samples_read_cleanly_and_export(samples: Path, tmp_path: Path) -> None:
    index = F.Index(samples)
    snapshot = index.rebuild()
    assert snapshot is not None, index.status().first_error
    assert snapshot.suites, "the samples form no suite"
    assert snapshot.runs, "the samples hold no run"
    assert [s.error for s in snapshot.sims if s.error] == []
    assert snapshot.judge_problems == ()
    E.export(snapshot, tmp_path / "gen")
