"""The version Release Please writes stays the same in every file it updates."""

from __future__ import annotations

import tomllib
from pathlib import Path

import corallab

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert corallab.__version__ == pyproject["project"]["version"]


def test_version_matches_citation() -> None:
    citation = (ROOT / "CITATION.cff").read_text()
    assert f"version: {corallab.__version__}" in citation
