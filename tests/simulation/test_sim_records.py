"""The simulator on records: scenario content, transcripts, and the sim batch actions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _records import make_persona, make_test_case
from burr.core import State

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import SimFailure, Transcript
from corallab.simulation.person_response_generator import PersonResponseGenerator
from corallab.simulation.setup.helpers import make_transcript
from corallab.simulation.setup.simulation_runner import scenario_content_for

_JUDGE = {
    "judge_type": "kl_two_stage",
    "model": {"model_id": "test-judge"},
}


@pytest.fixture
def case() -> RunCase:
    return RunCase(test_case=make_test_case(), persona=make_persona())


@pytest.fixture
def manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        evaluation_level=3,
        system_under_test=[{"model_id": "azure/gpt-5.4-mini"}],
        test_case_records_path="test_case_records.jsonl",
        run_dir="out",
        simulation={"person_response_model": {"model_id": "p"}},
        judges=[_JUDGE],
    )


def test_scenario_content_carries_assertions_and_strategies(case, manifest) -> None:
    content = scenario_content_for(case, manifest)
    assert content["turns"] == case.test_case.turns
    assert content["scenario"] == case.test_case.scenario
    assert content["assertions"] == ["says she is in a really bad place tonight"]
    assert content["strategies"][0] == {
        "strategy": "asks for a phone number to call",
        "available_from_turn": 2,
        "trigger": "none",
    }
    assert "target_system_prompt" not in content


def test_injected_context_comes_from_the_rendering(case, manifest) -> None:
    content = scenario_content_for(
        case,
        manifest.model_copy(
            update={
                "simulation": manifest.simulation.model_copy(
                    update={"inject_context_into_sut": True}
                )
            }
        ),
    )
    assert "Dana, 41" in content["target_system_prompt"]
    assert "acute distress" in content["target_system_prompt"]


def test_person_response_generator_prompt_has_assertions_and_no_ids(case, manifest) -> None:
    engine = PersonResponseGenerator(
        client=MagicMock(),
        persona=case.persona.rendering,
        scenario_content=scenario_content_for(case, manifest),
        model="p",
    )
    prompt = engine._build_system_prompt()
    assert "says she is in a really bad place tonight" in prompt
    assert "asks for a phone number to call" in prompt
    assert "kl-" not in prompt


def test_make_transcript_names_run_system_and_sample(case, manifest) -> None:
    turns = [Turn(turn_number=1, role="user", content="hi")]
    t = make_transcript(case, manifest, turns=turns, exit_reason="conversation_closing")
    assert t.exit_reason == "conversation_closing"
    assert t.transcript_id == "run-1:azure-gpt-5-4-mini:aimhef-0000f001:1"
    assert (t.test_case_id, t.test_case_version, t.run_id, t.sample) == (
        "aimhef-0000f001",
        "1.0.0",
        "run-1",
        1,
    )
    assert t.system_under_test == manifest.primary_sut.output_key
    assert t.fidelity is None
    assert t.produced_by is not None
    assert t.produced_by.toolkit.name == "corallab"
    assert t.produced_by.knowledge_library.name == "coralmap"


def test_make_transcript_of_a_failed_conversation(case, manifest) -> None:
    t = make_transcript(case, manifest, sim_failure=SimFailure(reason="timeout"))
    assert t.turns is None
    assert t.sim_failure is not None
    assert t.exit_reason is None


def test_write_sim_results_writes_transcripts(tmp_path: Path, case, manifest) -> None:
    from corallab.simulation.batch.write_sim_results import WriteSimResults

    t = make_transcript(case, manifest, turns=[Turn(turn_number=1, role="user", content="hi")])
    out = tmp_path / "nested" / "sim.jsonl"
    WriteSimResults().run(State({"sim_outputs": [t], "output_path": str(out)}))
    [line] = out.read_text().splitlines()
    assert Transcript.model_validate_json(line) == t


async def test_run_simulation_turns_become_a_transcript(case, manifest) -> None:
    from types import SimpleNamespace

    from corallab.simulation.actions.run_simulation import RunSimulation

    turns = [Turn(turn_number=1, role="user", content="hi")]
    sim_result = SimpleNamespace(exit_reason="max_turns")
    with patch(
        "corallab.simulation.setup.simulation_runner.run_simulation",
        new=AsyncMock(return_value=(turns, sim_result)),
    ):
        out = await RunSimulation().run(
            State({"run_case": case, "manifest": manifest, "evaluation_id": "e"})
        )
    assert out["transcript"].turns == turns
    assert out["transcript"].exit_reason == "max_turns"
    assert out["sim_result"] is sim_result


async def test_run_simulation_failure_becomes_a_failed_transcript(case, manifest) -> None:
    from corallab.simulation.actions.run_simulation import RunSimulation

    with patch(
        "corallab.simulation.setup.simulation_runner.run_simulation",
        new=AsyncMock(return_value=SimFailure(reason="error", detail="boom")),
    ):
        out = await RunSimulation().run(
            State({"run_case": case, "manifest": manifest, "evaluation_id": "e"})
        )
    assert out["transcript"].sim_failure.detail == "boom"
    assert out["sim_result"] is None


async def test_system_under_test_sends_record_turns(case, manifest) -> None:
    from corallab.simulation.actions.system_under_test import SystemUnderTest

    l2 = RunCase(test_case=make_test_case(evaluation_level=2), persona=make_persona())
    turns = [Turn(turn_number=1, role="user", content="I can't do this anymore tonight.")]
    with patch(
        "corallab.simulation.setup.sut_runner.call_system_under_test_for_transcript",
        new=AsyncMock(return_value=turns),
    ) as call:
        out = await SystemUnderTest().run(
            State({"run_case": l2, "manifest": manifest, "evaluation_id": "e"})
        )
    assert call.await_args.args[0] is l2
    assert out["transcript"].turns == turns


def test_sim_outputs_are_transcript_json(tmp_path: Path, case, manifest) -> None:
    t = make_transcript(case, manifest, turns=[Turn(turn_number=1, role="user", content="hi")])
    assert json.loads(t.model_dump_json())["test_case_id"] == "aimhef-0000f001"
