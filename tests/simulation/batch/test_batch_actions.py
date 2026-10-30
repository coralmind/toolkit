"""Tests for the sim batch actions: WriteSimResults and ParallelSim."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _records import make_persona, make_test_case
from burr.core import State
from burr.core.parallelism import RunnableGraph

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import SimFailure
from corallab.simulation.batch.parallel_sim import ParallelSim
from corallab.simulation.batch.write_sim_results import WriteSimResults
from corallab.simulation.setup.helpers import make_transcript

_JUDGE = {
    "judge_type": "kl_two_stage",
    "model": {"model_id": "test-judge"},
}


@pytest.fixture
def manifest() -> RunManifest:
    return RunManifest(
        run_id="test",
        evaluation_level=1,
        system_under_test=[{"model_id": "test-sut"}],
        test_case_records_path="",
        run_dir="",
        judges=[_JUDGE],
    )


def _case(test_case_id: str) -> RunCase:
    return RunCase(test_case=make_test_case(test_case_id=test_case_id), persona=make_persona())


class TestWriteSimResults:
    def test_writes_one_json_line_per_transcript(self, tmp_path: Path, manifest) -> None:
        turn = Turn(turn_number=1, role="user", content="hello")
        transcripts = [
            make_transcript(_case("aimhef-a1b2c3d4"), manifest, turns=[turn]),
            make_transcript(_case("aimhef-b2c3d4e5"), manifest, turns=[turn]),
        ]
        out_path = tmp_path / "sim_output.jsonl"

        result = WriteSimResults().run(
            State({"sim_outputs": transcripts, "output_path": str(out_path)})
        )

        assert result["results_written"] == 2
        ids = [json.loads(line)["test_case_id"] for line in out_path.read_text().splitlines()]
        assert ids == ["aimhef-a1b2c3d4", "aimhef-b2c3d4e5"]

    def test_empty_outputs_writes_empty_file(self, tmp_path: Path) -> None:
        out_path = tmp_path / "empty.jsonl"

        result = WriteSimResults().run(State({"sim_outputs": [], "output_path": str(out_path)}))

        assert result["results_written"] == 0
        assert out_path.read_text() == ""


class TestParallelSim:
    def test_states_yields_one_per_run_case(self, manifest) -> None:
        action = ParallelSim(MagicMock(spec=RunnableGraph))

        state = State(
            {
                "run_cases": [_case("aimhef-a1b2c3d4"), _case("aimhef-b2c3d4e5")],
                "manifest": manifest,
            }
        )
        yielded = list(action.states(state, MagicMock(), {}))

        assert [y["run_case"].test_case_id for y in yielded] == [
            "aimhef-a1b2c3d4",
            "aimhef-b2c3d4e5",
        ]
        assert yielded[0]["evaluation_level"] == 1
        assert yielded[0]["evaluation_id"].startswith("sim-")
        assert yielded[0]["transcript"] is None

    async def test_reduce_collects_transcripts_including_failures(self, manifest) -> None:
        ok = make_transcript(
            _case("aimhef-a1b2c3d4"),
            manifest,
            turns=[Turn(turn_number=1, role="user", content="hello")],
        )
        failed = make_transcript(
            _case("aimhef-b2c3d4e5"), manifest, sim_failure=SimFailure(reason="timeout")
        )
        action = ParallelSim(MagicMock(spec=RunnableGraph))

        async def _fake_results():
            yield State({"transcript": ok})
            yield State({"transcript": failed})
            yield State({"transcript": None})

        result_state = await action.reduce(State({"sim_outputs": []}), _fake_results())

        assert result_state["sim_outputs"] == [ok, failed]
