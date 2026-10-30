"""Viewer tests read a data repository and results tree made under ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    (root / "test_case_records").mkdir(parents=True)
    monkeypatch.setenv("CORALMIND_DATA_REPOSITORY_PATH", str(root))
    return root


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    root.mkdir()
    return root
