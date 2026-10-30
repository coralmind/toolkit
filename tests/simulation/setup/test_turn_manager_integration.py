"""Integration tests for the L3 simulation runner.

Covers:
  (a) Happy path: RunCase -> (turns, SimulationResult)
  (b) Simulation graph exception -> SimFailure
  (c) Empty turns list -> sim_result_to_turns raises ValueError
  (d) A manifest without person_response_model -> SimFailure
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from _records import make_persona, make_test_case

from corallab.schemas import RunManifest, SimulationResult, Turn
from corallab.schemas.eval.conversation_signal import ConversationSignal
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.transcript import SimFailure
from corallab.simulation.actions.sim_loop.call_sut import CallSUT
from corallab.simulation.person_response_generator import (
    PersonEchoError,
    PersonResponseGenerator,
)
from corallab.simulation.setup.simulation_runner import run_simulation, sim_result_to_turns

_JUDGE = {
    "judge_type": "kl_two_stage",
    "model": {"model_id": "claude-sonnet-4-6"},
}

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_case() -> RunCase:
    return RunCase(test_case=make_test_case(), persona=make_persona())


def _make_manifest() -> RunManifest:
    sim_cfg = {"model_id": "claude-sonnet-4-6"}
    return RunManifest(
        run_id="run-test-001",
        evaluation_level=3,
        system_under_test=[{"model_id": "claude-sonnet-4-6"}],
        test_case_records_path="tests/fixtures/input.jsonl",
        run_dir="/tmp/output",
        simulation={"person_response_model": sim_cfg},
        judges=[_JUDGE],
    )


def _make_sim_result(turns: list[Turn] | None = None) -> SimulationResult:
    if turns is None:
        turns = [
            Turn(turn_number=1, role="user", content="I need reassurance."),
            Turn(turn_number=2, role="assistant", content="I hear you."),
        ]
    return SimulationResult(
        test_case_id="aimhef-a1b2c3d4",
        run_config=RunManifest(
            run_id="test",
            evaluation_level=3,
            system_under_test=[{"model_id": "claude-sonnet-4-6"}],
            simulation={"person_response_model": {"model_id": "claude-sonnet-4-6"}},
            test_case_records_path="",
            run_dir="",
            judges=[_JUDGE],
        ),
        conversation_transcript=turns,
        exit_reason="conversation_closing",
        total_turns=2,
    )


# ---------------------------------------------------------------------------
# (c) sim_result_to_turns — empty transcript raises
# ---------------------------------------------------------------------------


def test_sim_result_to_turns_empty_raises():
    """Empty transcript must raise ValueError — not pass empty list to judge."""
    sim_result = _make_sim_result(turns=[])
    with pytest.raises(ValueError, match="empty transcript"):
        sim_result_to_turns(sim_result)


def test_sim_result_to_turns_happy_path():
    """Valid SimulationResult converts to list[Turn] correctly."""
    sim_result = _make_sim_result()
    turns = sim_result_to_turns(sim_result)

    assert len(turns) == 2
    assert turns[0].turn_number == 1
    assert turns[0].role == "user"
    assert turns[0].content == "I need reassurance."
    assert turns[1].role == "assistant"
    assert turns[1].content == "I hear you."


# ---------------------------------------------------------------------------
# (d) no person_response_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_simulation_without_person_response_model_returns_failure():
    # model_copy skips validation; the manifest validator rejects this at load time.
    manifest = _make_manifest().model_copy(
        update={
            "simulation": _make_manifest().simulation.model_copy(
                update={"person_response_model": None}
            )
        }
    )

    result = await run_simulation(_make_case(), manifest, "eval-test-001")

    assert isinstance(result, SimFailure)
    assert result.reason == "error"
    assert "simulation.person_response_model" in (result.detail or "")


# ---------------------------------------------------------------------------
# (b) Simulation graph exception -> SimFailure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_simulation_graph_exception_returns_failure():
    with patch(
        "corallab.simulation.setup.simulation_runner._run_sim_graph",
        new_callable=AsyncMock,
        side_effect=RuntimeError("API timeout"),
    ):
        result = await run_simulation(_make_case(), _make_manifest(), "eval-test-001")

    assert isinstance(result, SimFailure)
    assert result.reason == "error"
    assert result.detail == "API timeout"


@pytest.mark.asyncio
async def test_a_person_echo_ends_the_conversation_and_keeps_its_turns():
    person = AsyncMock(
        side_effect=[("and then", ConversationSignal()), PersonEchoError("copied the AI twice")]
    )
    with (
        patch.object(CallSUT, "_call_with_retry", new=AsyncMock(return_value="hi there")),
        patch.object(PersonResponseGenerator, "generate_message", new=person),
    ):
        result = await run_simulation(_make_case(), _make_manifest(), "eval-test-001")

    assert isinstance(result, tuple)
    turns, sim_result = result
    assert sim_result.exit_reason == "person_echo"
    assert [(t.turn_number, t.role) for t in turns] == [
        (1, "user"),
        (1, "assistant"),
        (2, "user"),
        (2, "assistant"),
    ]


# ---------------------------------------------------------------------------
# (a) Happy path: simulation graph succeeds -> returns (turns, sim_result)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_simulation_happy_path_returns_turns_and_result():
    sim_result = _make_sim_result()

    with patch(
        "corallab.simulation.setup.simulation_runner._run_sim_graph",
        new_callable=AsyncMock,
        return_value=sim_result,
    ):
        result = await run_simulation(_make_case(), _make_manifest(), "eval-test-001")

    assert isinstance(result, tuple)
    turns, returned_sim_result = result
    assert len(turns) == 2
    assert returned_sim_result.exit_reason == "conversation_closing"
