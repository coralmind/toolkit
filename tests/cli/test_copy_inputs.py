"""Tests for input provenance copying in ``corallab sim``/``judge``/``run``.

Issue #59 — until a database backend lands (#207), every run freezes a copy of
the exact run manifest and the test-case JSONL it was driven by into
``<out_dir>/inputs/``. The batch runners and report renderers are monkeypatched
to canned no-ops (no Burr graph, no LLM calls), so these tests only exercise the
CLI wiring around ``copy_run_inputs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corallab.cli import app

runner = CliRunner()

TC_ID = "aimhef-0a1b2c3d"


def _rec() -> dict:
    return {
        "transcript_id": f"sim-view-test:azure-gpt-4-1-mini:{TC_ID}:1",
        "test_case_id": TC_ID,
        "test_case_version": "1.0.0",
        "run_id": "sim-view-test",
        "system_under_test": "azure/gpt-4.1-mini",
        "sample": 1,
        "turns": [
            {"turn_number": 1, "role": "user", "content": "hello"},
            {"turn_number": 2, "role": "assistant", "content": "hi there"},
        ],
    }


def _write_manifest(tmp_path: Path, **overrides: object) -> Path:
    data = {
        "run_id": "copy-inputs-test",
        "evaluation_level": 1,
        "system_under_test": [{"model_id": "azure/gpt-4.1-mini"}],
        "test_case_records_path": "in.jsonl",
        "run_dir": str(tmp_path / "results" / "out"),
        "judges": [
            {
                "judge_type": "kl_two_stage",
                "model": {"model_id": "azure/gpt-4.1-mini"},
            }
        ],
    }
    data.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in.jsonl").write_text(json.dumps(_rec()) + "\n")

    def _write_fake_output(out_path: Path) -> None:
        Path(out_path).write_text(json.dumps(_rec()) + "\n")

    async def fake_run_batch_sim(manifest, input_path, out_path, db_path, run_stem, dry_run=False):
        from corallab.llm_client import TokenAccumulator

        _write_fake_output(Path(out_path))
        return [_rec()], [], TokenAccumulator(), [Path(out_path)]

    async def fake_run_batch_judge(
        manifest, transcripts_path, out_path, db_path, run_stem, dry_run=False
    ):
        from corallab.llm_client import TokenAccumulator

        _write_fake_output(Path(out_path))
        return [_rec()], TokenAccumulator()

    monkeypatch.setattr("corallab.cli._runners.run_batch_sim", fake_run_batch_sim)
    monkeypatch.setattr("corallab.cli._runners.run_batch_judge", fake_run_batch_judge)
    return tmp_path


def _inputs_dir() -> Path:
    matches = list(Path("results").rglob("inputs"))
    assert len(matches) == 1, f"expected exactly one inputs/ dir, got {matches}"
    return matches[0]


def test_sim_copies_manifest_and_test_cases(workspace: Path) -> None:
    manifest = _write_manifest(workspace)
    result = runner.invoke(app, ["sim", "--run-manifest", str(manifest)])
    assert result.exit_code == 0, result.output

    inputs = _inputs_dir()
    assert (inputs / "manifest.json").exists()
    assert (inputs / "in.jsonl").exists()
    # Copies are byte-identical to the originals.
    assert (inputs / "manifest.json").read_text() == manifest.read_text()
    assert (inputs / "in.jsonl").read_text() == (workspace / "in.jsonl").read_text()


def test_judge_leaves_the_runs_inputs_alone(workspace: Path) -> None:
    # Judge writes into the run that already holds the transcripts and does not open
    # another folder. The inputs are the simulation's record; judging does not replace them.
    manifest = _write_manifest(workspace)
    run_dir = workspace / "results" / "out-20261007-120000"
    batch = run_dir / "batch-001"
    batch.mkdir(parents=True)
    transcripts = batch / "out-20261007-120000-sim.jsonl"
    transcripts.write_text(json.dumps(_rec()) + "\n")
    inputs = run_dir / "inputs"
    inputs.mkdir()
    (inputs / "in.jsonl").write_text("as simulated\n")
    result = runner.invoke(
        app, ["judge", "--run-manifest", str(manifest), "--transcripts", str(transcripts)]
    )
    assert result.exit_code == 0, result.output

    assert sorted(p.name for p in inputs.iterdir()) == ["in.jsonl"]
    assert (inputs / "in.jsonl").read_text() == "as simulated\n"
    assert [p.resolve() for p in Path("results").rglob("inputs")] == [inputs]


def test_run_copies_inputs_once(workspace: Path) -> None:
    # The full pipeline copies inputs a single time into the shared run folder,
    # not once per stage.
    manifest = _write_manifest(workspace)
    result = runner.invoke(app, ["run", "--run-manifest", str(manifest)])
    assert result.exit_code == 0, result.output

    inputs = _inputs_dir()
    assert (inputs / "manifest.json").exists()
    assert (inputs / "in.jsonl").exists()


def test_missing_test_cases_does_not_abort_run(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A copy failure (e.g. the test-case file is gone) is logged but must not
    # fail the command — preserving inputs is a convenience, not core.
    (workspace / "in.jsonl").unlink()
    manifest = _write_manifest(workspace)
    result = runner.invoke(app, ["sim", "--run-manifest", str(manifest)])
    assert result.exit_code == 0, result.output

    inputs = _inputs_dir()
    assert (inputs / "manifest.json").exists()
    assert not (inputs / "in.jsonl").exists()
