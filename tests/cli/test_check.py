"""``corallab check`` writes one fidelity check record per transcript beside the transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _records import make_persona, make_test_case, write_test_case_records
from typer.testing import CliRunner

from corallab.cli import app
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import FidelityCheckRecord
from corallab.simulation.setup.helpers import make_transcript

runner = CliRunner()
_CHECK = "corallab.fidelity.elicitation.ElicitationFidelityChecker.check"


def _run(tmp_path: Path, **fidelity: object) -> tuple[Path, Path]:
    """A manifest and one batch of transcripts in a sim run folder."""
    records_path, personas = write_test_case_records(tmp_path, [make_test_case()], [make_persona()])
    manifest = {
        "run_id": "e2e",
        "evaluation_level": 3,
        "system_under_test": [{"model_id": "mock/sut"}],
        "test_case_records_path": str(records_path),
        "persona_records_path": str(personas),
        "run_dir": str(tmp_path / "results" / "e2e"),
        "simulation": {
            "person_response_model": {"model_id": "mock/persona"},
            "elicitation_fidelity": fidelity,
        },
        "judges": [{"judge_type": "kl_two_stage", "model": {"model_id": "mock/judge"}}],
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    case = RunCase(test_case=make_test_case(), persona=make_persona())
    transcript = make_transcript(
        case,
        RunManifest.model_validate(manifest),
        turns=[
            Turn(turn_number=1, role="user", content="I'm in a really bad place tonight."),
            Turn(turn_number=1, role="assistant", content="I'm here. What happened?"),
        ],
    )
    batch = tmp_path / "results" / "e2e-20261007-120000" / "batch-001"
    batch.mkdir(parents=True)
    tpath = batch / "e2e-20261007-120000-sim.jsonl"
    tpath.write_text(transcript.model_dump_json() + "\n")
    return mpath, tpath


def test_check_writes_a_record_beside_the_transcripts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mpath, tpath = _run(tmp_path, model={"model_id": "mock/checker"})
    with patch(_CHECK, new=AsyncMock(return_value=None)):
        result = runner.invoke(
            app, ["check", "--run-manifest", str(mpath), "--transcripts", str(tpath)]
        )
    assert result.exit_code == 0, result.output
    out = tpath.with_name("e2e-20261007-120000-check.jsonl")
    (line,) = out.read_text().splitlines()
    record = FidelityCheckRecord.model_validate_json(line)
    assert record.transcript_id == "e2e:mock-sut:aimhef-0000f001:1"
    assert record.fidelity is None
    assert record.error is not None
    assert json.loads(tpath.read_text())["fidelity"] is None
    assert "simulator's own model" not in result.output


def test_check_reads_the_test_cases_the_run_was_simulated_from(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mpath, tpath = _run(tmp_path, model={"model_id": "mock/checker"})
    records = Path(json.loads(mpath.read_text())["test_case_records_path"])
    record = json.loads(records.read_text().splitlines()[0])
    record["scenario_strategies"][0]["strategy"] = "the strategy the run played"
    inputs = tpath.parent.parent / "inputs"
    inputs.mkdir()
    (inputs / records.name).write_text(json.dumps(record) + "\n")
    with patch(_CHECK, new=AsyncMock(return_value=None)) as checked:
        result = runner.invoke(
            app, ["check", "--run-manifest", str(mpath), "--transcripts", str(tpath)]
        )
    assert result.exit_code == 0, result.output
    strategy = checked.call_args.kwargs["test_case"].scenario_strategies[0].strategy
    assert strategy == "the strategy the run played"


def test_check_warns_when_the_simulator_checks_itself(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mpath, tpath = _run(tmp_path)
    with patch(_CHECK, new=AsyncMock(return_value=None)):
        result = runner.invoke(
            app, ["check", "--run-manifest", str(mpath), "--transcripts", str(tpath)]
        )
    assert result.exit_code == 0, result.output
    assert "simulator's own model, mock/persona" in " ".join(result.output.split())


@pytest.mark.parametrize("version", ["elicitation-fidelity-v2", "elicitation-fidelity-missing"])
def test_check_refuses_a_prompt_it_cannot_read(tmp_path: Path, monkeypatch, version: str) -> None:
    monkeypatch.chdir(tmp_path)
    mpath, tpath = _run(tmp_path, prompt_version=version)
    with patch(_CHECK, new=AsyncMock()) as checked:
        result = runner.invoke(
            app, ["check", "--run-manifest", str(mpath), "--transcripts", str(tpath)]
        )
    assert result.exit_code == 1
    checked.assert_not_called()
    assert not tpath.with_name("e2e-20261007-120000-check.jsonl").exists()


def test_run_checks_each_batch_before_judging(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    mpath, tpath = _run(tmp_path, model={"model_id": "mock/checker"})
    transcript = tpath.read_text()

    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    async def fake_sim(manifest, input_path, out_path, db_path, run_stem, dry_run=False):
        from corallab.llm_client import TokenAccumulator

        written = Path(out_path).parent / "batch-001" / Path(out_path).name
        write(written, transcript)
        return [json.loads(transcript)], [], TokenAccumulator(), [written]

    async def fake_judge(manifest, transcripts_path, out_path, db_path, run_stem, dry_run=False):
        from corallab.llm_client import TokenAccumulator

        write(Path(out_path), "{}\n")
        return [{}], TokenAccumulator()

    monkeypatch.setattr("corallab.cli._runners.run_batch_sim", fake_sim)
    monkeypatch.setattr("corallab.cli._runners.run_batch_judge", fake_judge)
    with patch(_CHECK, new=AsyncMock(return_value=None)):
        result = runner.invoke(app, ["run", "--run-manifest", str(mpath)])
    assert result.exit_code == 0, result.output
    (batch,) = (b for b in (tmp_path / "results").glob("e2e-*/batch-001") if b != tpath.parent)
    assert sorted(p.name.rsplit("-", 1)[-1] for p in batch.glob("*.jsonl")) == [
        "check.jsonl",
        "judge.jsonl",
        "sim.jsonl",
    ]
