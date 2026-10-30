"""The transcript record and its fidelity verdict: which verdicts the models accept."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from corallab.schemas.eval.transcript import (
    BehaviorVerdict,
    FidelityVerdict,
    StrategyVerdict,
    Transcript,
)

BEHAVIOR = "kl-000410"


def _cite(turn: int = 1, speaker: str = "user") -> dict[str, Any]:
    return {"turn": turn, "speaker": speaker, "quote": "is it okay if I just don't go?"}


def _occurred() -> dict[str, Any]:
    return {
        "behavior": BEHAVIOR,
        "catalog_occurred": True,
        "assertion_occurred": True,
        "citations": [_cite()],
    }


def _verdict(**overrides: Any) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "checker": {"model": "m", "prompt_version": "v1"},
        "behaviors": [_occurred()],
        "strategies": [],
        "status": "valid",
        "reasons": [],
    }
    verdict.update(overrides)
    return verdict


def _transcript(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "transcript_id": "t-1",
        "test_case_id": "aimhef-0000abcd",
        "test_case_version": "1.0.0",
        "run_id": "run-1",
        "system_under_test": "sut",
        "sample": 1,
        "turns": [{"turn_number": 1, "role": "user", "content": "hello"}],
    }
    record.update(overrides)
    return record


class TestFidelityStatus:
    def test_every_behavior_occurring_is_valid(self) -> None:
        assert FidelityVerdict.model_validate(_verdict()).status == "valid"

    def test_a_status_that_does_not_follow_is_rejected(self) -> None:
        missed = {
            "behavior": BEHAVIOR,
            "catalog_occurred": False,
            "assertion_occurred": False,
            "miss": "not_attempted",
        }
        with pytest.raises(ValidationError, match="status should be invalid"):
            FidelityVerdict.model_validate(_verdict(behaviors=[missed]))

    def test_an_invalid_transcript_names_each_missed_behavior(self) -> None:
        missed = {
            "behavior": BEHAVIOR,
            "catalog_occurred": False,
            "assertion_occurred": False,
            "miss": "not_attempted",
        }
        other = {
            "behavior": "kl-000411",
            "catalog_occurred": False,
            "assertion_occurred": False,
            "miss": "not_attempted",
        }
        verdict = FidelityVerdict.model_validate(
            _verdict(
                behaviors=[missed, other],
                status="invalid",
                reasons=[
                    {"behavior": BEHAVIOR, "miss": "not_attempted"},
                    {"behavior": "kl-000411", "miss": "not_attempted"},
                ],
            )
        )
        assert [item.behavior for item in verdict.reasons] == [BEHAVIOR, "kl-000411"]


class TestBehaviorVerdict:
    def test_an_occurred_behavior_cites_turns_and_has_no_miss(self) -> None:
        with pytest.raises(ValidationError, match="citations and no miss"):
            BehaviorVerdict.model_validate({**_occurred(), "citations": []})

    def test_a_missed_behavior_names_its_miss(self) -> None:
        with pytest.raises(ValidationError, match="has a miss class"):
            BehaviorVerdict.model_validate(
                {"behavior": BEHAVIOR, "catalog_occurred": False, "assertion_occurred": False}
            )


class TestStrategyVerdict:
    def test_an_absent_trigger_cites_nothing(self) -> None:
        with pytest.raises(ValidationError, match="absent trigger cites nothing"):
            StrategyVerdict.model_validate(
                {"index": 1, "adherence": "trigger_absent", "citations": [_cite()]}
            )

    def test_a_followed_strategy_cites_the_user_turn(self) -> None:
        with pytest.raises(ValidationError, match="cites the user turn"):
            StrategyVerdict.model_validate({"index": 1, "adherence": "followed"})


class TestTranscript:
    def test_a_conversation_with_its_verdict(self) -> None:
        record = Transcript.model_validate(_transcript(fidelity=_verdict()))
        assert record.fidelity is not None
        assert record.fidelity.status == "valid"

    def test_a_transcript_has_turns_or_a_failure_not_both(self) -> None:
        failure = {"reason": "timeout"}
        assert Transcript.model_validate(_transcript(turns=None, sim_failure=failure))
        with pytest.raises(ValidationError, match="either turns or a sim_failure"):
            Transcript.model_validate(_transcript(sim_failure=failure))
        with pytest.raises(ValidationError, match="either turns or a sim_failure"):
            Transcript.model_validate(_transcript(turns=None))

    def test_a_failed_run_has_no_fidelity_verdict(self) -> None:
        with pytest.raises(ValidationError, match="no fidelity verdict"):
            Transcript.model_validate(
                _transcript(turns=None, sim_failure={"reason": "timeout"}, fidelity=_verdict())
            )

    def test_exit_reason_is_recorded_on_a_finished_conversation(self) -> None:
        record = Transcript.model_validate(_transcript(exit_reason="max_turns"))
        assert record.exit_reason == "max_turns"
        assert Transcript.model_validate(_transcript()).exit_reason is None

    def test_a_failed_run_has_no_exit_reason(self) -> None:
        with pytest.raises(ValidationError, match="no exit_reason"):
            Transcript.model_validate(
                _transcript(turns=None, sim_failure={"reason": "timeout"}, exit_reason="max_turns")
            )

    def test_an_unknown_exit_reason_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Transcript.model_validate(_transcript(exit_reason="scenario_enacted"))

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            Transcript.model_validate(_transcript(notes="x"))

    def test_an_exchange_shares_one_turn_number(self) -> None:
        record = Transcript.model_validate(
            _transcript(
                turns=[
                    {"turn_number": 1, "role": "user", "content": "hello"},
                    {"turn_number": 1, "role": "assistant", "content": "hi"},
                    {"turn_number": 2, "role": "user", "content": "and then"},
                    {"turn_number": 2, "role": "assistant", "content": "go on"},
                ]
            )
        )
        assert [turn.turn_number for turn in record.turns or []] == [1, 1, 2, 2]

    def test_an_assistant_reply_on_the_next_number_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="does not answer user exchange 1"):
            Transcript.model_validate(
                _transcript(
                    turns=[
                        {"turn_number": 1, "role": "user", "content": "hello"},
                        {"turn_number": 2, "role": "assistant", "content": "hi"},
                    ]
                )
            )

    def test_a_user_message_that_skips_an_exchange_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exchange 2 is next"):
            Transcript.model_validate(
                _transcript(
                    turns=[
                        {"turn_number": 1, "role": "user", "content": "hello"},
                        {"turn_number": 1, "role": "assistant", "content": "hi"},
                        {"turn_number": 3, "role": "user", "content": "later"},
                    ]
                )
            )

    def test_a_repeated_message_in_an_exchange_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="turn 1 user is repeated"):
            Transcript.model_validate(
                _transcript(
                    turns=[
                        {"turn_number": 1, "role": "user", "content": "hello"},
                        {"turn_number": 1, "role": "user", "content": "again"},
                    ]
                )
            )

    def test_an_assistant_message_with_no_user_message_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="has no user message"):
            Transcript.model_validate(
                _transcript(turns=[{"turn_number": 1, "role": "assistant", "content": "hi"}])
            )

    def test_a_turn_number_below_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Transcript.model_validate(
                _transcript(turns=[{"turn_number": 0, "role": "user", "content": "hello"}])
            )

    def test_an_unknown_role_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="user"):
            Transcript.model_validate(
                _transcript(turns=[{"turn_number": 1, "role": "system", "content": "hello"}])
            )
