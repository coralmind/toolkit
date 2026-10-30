"""The results viewer pins the brand package and finds results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corallab.paths import default_repo_root, resolve_run_dir
from coralmind_viewer.catalog import scan

VIEWER = default_repo_root() / "viewer"


def test_the_brand_is_the_npm_package_the_kl_viewer_pins() -> None:
    package = json.loads((VIEWER / "package.json").read_text(encoding="utf-8"))
    pin = package["dependencies"]["@coralmind/brand"]
    assert pin.startswith("git+ssh://git@github.com/coralmind/brand.git#")
    lock = json.loads((VIEWER / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"]["node_modules/@coralmind/brand"]["resolved"] == pin


def test_relative_run_dir_lands_under_the_data_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORALMIND_DATA_REPOSITORY_PATH", str(tmp_path))
    resolved = resolve_run_dir("results/l3-launch-entrenchment-small-azure")
    assert resolved == tmp_path / "results" / "l3-launch-entrenchment-small-azure"


def test_absolute_run_dir_is_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    assert resolve_run_dir(target) == target


def test_scan_lists_a_run_batch_and_its_jsonl(tmp_path: Path) -> None:
    batch = tmp_path / "results" / "demo" / "batch-001"
    batch.mkdir(parents=True)
    (batch / "demo-sim.jsonl").write_text(json.dumps({"test_case_id": "x"}) + "\n")
    (batch / "demo-judge.jsonl").write_text(json.dumps({"test_case_id": "x"}) + "\n")
    runs = scan(tmp_path / "results")
    assert [run.name for run in runs] == ["demo"]
    found = runs[0].batches[0]
    assert found.name == "batch-001"
    assert found.sim_files[0].name == "demo-sim.jsonl"
    assert found.judge_files[0].name == "demo-judge.jsonl"
