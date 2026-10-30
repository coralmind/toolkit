"""``corallab sim`` writes transcripts and does not generate HTML."""

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


@pytest.fixture
def manifest_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)

    def _write_fake_output(out_path: Path) -> None:
        out_path.write_text(json.dumps(_rec()) + "\n")

    async def fake_run_batch_sim(manifest, input_path, out_path, db_path, run_stem, dry_run=False):
        from corallab.llm_client import TokenAccumulator

        _write_fake_output(Path(out_path))
        return [_rec()], [], TokenAccumulator(), [Path(out_path)]

    monkeypatch.setattr("corallab.cli._runners.run_batch_sim", fake_run_batch_sim)

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "sim-view-test",
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
        )
    )
    return path


def test_sim_writes_transcripts_without_html(manifest_path: Path) -> None:
    result = runner.invoke(app, ["sim", "--run-manifest", str(manifest_path)])
    assert result.exit_code == 0, result.output
    transcripts = list(Path("results").rglob("*-sim.jsonl"))
    assert transcripts
    assert "hi there" in transcripts[0].read_text()
    assert list(Path("results").rglob("*.html")) == []


def test_sim_fanout_writes_one_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two models produce one index and one transcript file each, and no HTML."""
    monkeypatch.chdir(tmp_path)

    def _write_fake_output(out_path: Path) -> None:
        out_path.write_text(json.dumps(_rec()) + "\n")

    async def fake_run_batch_sim(manifest, input_path, out_path, db_path, run_stem, dry_run=False):
        from corallab.llm_client import TokenAccumulator

        _write_fake_output(Path(out_path))
        return [_rec()], [], TokenAccumulator(), [Path(out_path)]

    monkeypatch.setattr("corallab.cli._runners.run_batch_sim", fake_run_batch_sim)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "fanout-test",
                "evaluation_level": 1,
                "system_under_test": [
                    {"model_id": "azure/gpt-4.1-mini", "label": "gpt41"},
                    {"model_id": "gemini/gemini-2.5-flash", "label": "gemini"},
                ],
                "test_case_records_path": "in.jsonl",
                "run_dir": str(tmp_path / "results" / "out"),
                "judges": [
                    {
                        "judge_type": "kl_two_stage",
                        "model": {"model_id": "azure/gpt-4.1-mini"},
                    }
                ],
            }
        )
    )

    result = runner.invoke(app, ["sim", "--run-manifest", str(manifest)])
    assert result.exit_code == 0, result.output

    indexes = list(Path("results").rglob("*.index.json"))
    assert len(indexes) == 1
    index = indexes[0].read_text()
    assert "azure/gpt-4.1-mini" in index
    assert "gemini/gemini-2.5-flash" in index
    assert len(list(Path("results").rglob("*-sim.jsonl"))) == 2
    assert list(Path("results").rglob("*.html")) == []
