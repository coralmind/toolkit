"""A transcript record: one simulated conversation of a test case, and whether it enacted it.

A test case run several times against several systems has one transcript record per
conversation, naming the test case, the run, the system under test and the sample.
Its ``fidelity`` verdict says whether the simulated person did what the test case required,
and so whether the conversation may be judged: only a ``valid`` transcript is.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from corallab.schemas.core.code_version import ProducedBy
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.provenance import KL_ID, TEST_CASE_ID

# Why an L3 conversation stopped. Checked after each system-under-test reply:
# the turn cap first, then the persona's closing signal. ``scenario_enacted``
# does not stop the loop.
ConversationEndReason = Literal["max_turns", "conversation_closing", "person_echo"]


class SimFailure(BaseModel):
    """Why a simulation failed, so the test case stays in the output with no turns."""

    reason: str = Field(
        description=(
            "Failure category: 'content_filter' (provider blocked the prompt — "
            "deterministic, not retryable), 'rate_limit', 'timeout', "
            "'empty_response', or 'error' (anything else)."
        )
    )
    detail: str | None = Field(
        default=None, description="Human-readable error message (truncated)."
    )
    failed_at_turn: int | None = Field(
        default=None, description="1-indexed user turn whose SUT response failed, if known."
    )


class TurnCitation(BaseModel):
    """A verbatim quote from one turn of a transcript."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(
        ge=1,
        description="1-indexed exchange. speaker says which message of that exchange",
    )
    speaker: Literal["user", "assistant"] = Field(
        default="user",
        description=(
            "Who spoke this message of the exchange. Required on a fidelity citation: "
            "the user message and the assistant reply share turn"
        ),
    )
    quote: str = Field(min_length=1, description="The quoted words, verbatim")


class PartVerdict(BaseModel):
    """Whether one lettered part of a behavior's criteria showed up."""

    model_config = ConfigDict(extra="forbid")

    part: str = Field(description="The part's letter in the criteria, or 'all' when unlettered")
    present: bool
    citations: list[TurnCitation] = Field(
        default_factory=list, description="The user turns that show the part; empty when absent"
    )
    ruled_out_by: TurnCitation | None = Field(
        default=None,
        description="The user turn that breaks a part not present; null when no turn does",
    )
    reason: str | None = Field(
        default=None,
        description=(
            "The checker's one-sentence account: how the citations show the part, what in "
            "ruled_out_by breaks it, or what never happened. Not checked against the transcript"
        ),
    )

    @model_validator(mode="after")
    def _ruled_out_only_when_not_present(self) -> PartVerdict:
        if self.present and self.ruled_out_by:
            raise ValueError(f"part {self.part}: a present part is not ruled out")
        if self.ruled_out_by and not self.reason:
            raise ValueError(f"part {self.part}: a ruled-out part says what breaks it")
        return self


class BehaviorVerdict(BaseModel):
    """Whether one expected behavior occurred."""

    model_config = ConfigDict(extra="forbid")

    behavior: str = Field(pattern=KL_ID, description="The expected user behavior")
    catalog_occurred: bool = Field(
        description=(
            "The behavior occurred as the catalog entry defines it, within its detection window. "
            "Decides validity"
        )
    )
    assertion_occurred: bool = Field(
        description="The behavior occurred in the form the test case's assertion states"
    )
    parts: list[PartVerdict] = Field(
        default_factory=list, description="One per lettered part of the criteria, in order"
    )
    citations: list[TurnCitation] = Field(
        default_factory=list,
        description=(
            "The user turns the behavior occurred in; every turn a multi-turn pattern spans"
        ),
    )
    miss: Literal["not_attempted", "preempted"] | None = Field(
        default=None,
        description=(
            "Set when the behavior did not occur. 'not_attempted': the simulated person did not "
            "do it, though nothing stood in the way. 'preempted': a multi-turn behavior the AI's "
            "earlier turns made unnecessary or impossible"
        ),
    )
    preempted_by: Literal["resolved", "derailed"] | None = Field(
        default=None,
        description=(
            "For a preempted miss. 'resolved': the AI met the need, so the behavior had no "
            "occasion. 'derailed': the AI misread the situation, so the scenario never happened"
        ),
    )
    ai_citation: TurnCitation | None = Field(
        default=None,
        description="For a preempted miss: the AI turn that made the behavior unnecessary",
    )
    strategy_index: int | None = Field(
        default=None,
        ge=1,
        description="For a preempted miss: the strategy whose trigger the AI's turn kept away",
    )

    @model_validator(mode="after")
    def _consistent(self) -> BehaviorVerdict:
        if self.catalog_occurred and (self.miss or not self.citations):
            raise ValueError(f"{self.behavior}: an occurred behavior has citations and no miss")
        if not self.catalog_occurred and self.miss is None:
            raise ValueError(f"{self.behavior}: a behavior that did not occur has a miss class")
        preempted = self.miss == "preempted"
        if preempted != (self.preempted_by is not None) or preempted != (
            self.ai_citation is not None
        ):
            raise ValueError(
                f"{self.behavior}: preempted_by and ai_citation go with a preempted miss only"
            )
        if self.ai_citation is not None and self.ai_citation.speaker != "assistant":
            raise ValueError(f"{self.behavior}: ai_citation quotes the assistant")
        return self


class StrategyVerdict(BaseModel):
    """Whether the person followed one scenario strategy."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(
        ge=1,
        description="1-indexed position of the strategy in the test case's scenario_strategies",
    )
    adherence: Literal["trigger_absent", "followed", "not_followed"] = Field(
        description=(
            "'trigger_absent': the trigger never happened. 'followed': the person acted on it. "
            "'not_followed': the trigger happened, or was 'none', and the person did not act on it"
        )
    )
    citations: list[TurnCitation] = Field(
        default_factory=list,
        description=(
            "For 'followed', the user turn; for 'not_followed', the turn that was the trigger, "
            "where there was one"
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "The checker's one-sentence account of the adherence; null when code decided it "
            "differently from the checker. Not checked against the transcript"
        ),
    )

    @model_validator(mode="after")
    def _consistent(self) -> StrategyVerdict:
        if self.adherence == "trigger_absent" and self.citations:
            raise ValueError(f"strategy {self.index}: an absent trigger cites nothing")
        if self.adherence == "followed" and not self.citations:
            raise ValueError(f"strategy {self.index}: a followed strategy cites the user turn")
        return self


class FidelityReason(BaseModel):
    """One catalog behavior whose absence makes the transcript invalid."""

    model_config = ConfigDict(extra="forbid")

    behavior: str = Field(pattern=KL_ID, description="The catalog behavior that did not occur")
    miss: Literal["not_attempted"] = Field(
        description="The behavior did not show up in the transcript"
    )


class FidelityChecker(BaseModel):
    """The model and prompt that produced a fidelity verdict."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model id of the checker")
    prompt_version: str = Field(description="Version of the checker's prompt")


class FidelityVerdict(BaseModel):
    """Whether a transcript enacted its test case, and so whether it may be judged.

    ``status`` follows from the behaviors: ``invalid`` when any behavior was not attempted;
    otherwise ``not_tested`` when any was preempted; otherwise ``valid``. Only a ``valid``
    transcript is judged.
    """

    model_config = ConfigDict(extra="forbid")

    checker: FidelityChecker
    behaviors: list[BehaviorVerdict] = Field(
        min_length=1, description="One per expected behavior of the test case"
    )
    strategies: list[StrategyVerdict] = Field(description="One per strategy of the test case")
    status: Literal["valid", "invalid", "not_tested"]
    reasons: list[FidelityReason] = Field(
        description=(
            "One entry per catalog behavior that did not occur, in behavior order. "
            "Empty unless status is 'invalid'"
        )
    )

    @model_validator(mode="after")
    def _status_follows(self) -> FidelityVerdict:
        reasons = [
            FidelityReason(behavior=item.behavior, miss="not_attempted")
            for item in self.behaviors
            if item.miss == "not_attempted"
        ]
        status = "invalid" if reasons else "valid"
        if status == "valid" and any(item.miss == "preempted" for item in self.behaviors):
            status = "not_tested"
        if self.status != status or list(self.reasons) != reasons:
            raise ValueError(f"status should be {status} with reasons {reasons}")
        return self


class Transcript(BaseModel):
    """One simulated conversation of one test case, in one run, against one system under test.

    A test case run *K* times against several systems has one record per conversation.
    The judge reads a transcript whose ``fidelity.status`` is 'valid', with the test case
    it names and the run's fact modes.
    """

    model_config = ConfigDict(extra="forbid")

    transcript_id: str = Field(
        description=(
            "Unique id of this conversation: <run_id>:<system_under_test>:<test_case_id>:<sample>"
        )
    )
    test_case_id: str = Field(pattern=TEST_CASE_ID, description="The test case simulated")
    test_case_version: str = Field(description="The version of the test case simulated")
    run_id: str = Field(
        description="The run manifest's run_id, which fixes the simulator and the judge"
    )
    system_under_test: str = Field(
        description="Output key of the run's system_under_test entry that answered"
    )
    sample: int = Field(
        ge=1, description="1-indexed sample number of this conversation within its run"
    )
    turns: list[Turn] | None = Field(
        default=None,
        description=(
            "The conversation, in order; null when the run failed. Exchanges are numbered "
            "from 1. Each exchange is a user message and, when the reply arrived, the "
            "assistant message with the same turn_number"
        ),
    )
    exit_reason: ConversationEndReason | None = Field(
        default=None,
        description=(
            "Why an L3 conversation stopped, after the system under test's last reply. "
            "'max_turns': round_count reached simulation.max_turns (this wins when the persona "
            "also signaled a close on that same reply). 'conversation_closing': the persona's "
            "previous message had nothing left to say, and the turn cap had not been reached. "
            "'person_echo': the simulated person copied the AI's message and copied it again "
            "when asked once more, so the copy was never sent and the conversation ends at the "
            "AI's previous reply. Null for a failed run and for an L1 or L2 scripted call, which "
            "has no loop"
        ),
    )
    sim_failure: SimFailure | None = Field(
        default=None, description="Set when the system under test failed and no conversation exists"
    )
    fidelity: FidelityVerdict | None = Field(
        default=None, description="The fidelity check's verdict; null until checked"
    )
    produced_by: ProducedBy | None = Field(
        default=None,
        description=(
            "Toolkit and Knowledge Library versions that wrote this transcript. "
            "Null on a record written before this was recorded"
        ),
    )

    @model_validator(mode="after")
    def _one_outcome(self) -> Transcript:
        if (self.turns is None) == (self.sim_failure is None):
            raise ValueError("a transcript has either turns or a sim_failure")
        if self.fidelity is not None and self.turns is None:
            raise ValueError("a failed run has no fidelity verdict")
        if self.exit_reason is not None and self.turns is None:
            raise ValueError("a failed run has no exit_reason")
        if self.turns:
            _check_exchanges(self.turns)
        return self


def _check_exchanges(turns: list[Turn]) -> None:
    """Exchanges are 1, 2, 3, … in order. Each is one user message, then its reply."""
    seen: set[tuple[int, str]] = set()
    next_exchange = 1
    open_exchange: int | None = None
    for turn in turns:
        key = (turn.turn_number, turn.role)
        if key in seen:
            raise ValueError(f"turn {turn.turn_number} {turn.role} is repeated")
        seen.add(key)
        if turn.role == "user":
            if open_exchange is not None:
                open_exchange = None
            if turn.turn_number != next_exchange:
                raise ValueError(
                    f"user message {turn.turn_number} is out of order; "
                    f"exchange {next_exchange} is next"
                )
            open_exchange = turn.turn_number
            next_exchange += 1
            continue
        if open_exchange is None:
            raise ValueError(
                f"assistant message {turn.turn_number} has no user message in that exchange"
            )
        if open_exchange != turn.turn_number:
            raise ValueError(
                f"assistant message {turn.turn_number} does not answer "
                f"user exchange {open_exchange}"
            )
        open_exchange = None


class FidelityCheckRecord(BaseModel):
    """One line of a fidelity check's output: its verdict on one transcript."""

    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    test_case_id: str = Field(pattern=TEST_CASE_ID)
    system_under_test: str
    fidelity: FidelityVerdict | None = Field(
        default=None, description="The verdict; null when the check could not be completed"
    )
    error: str | None = Field(default=None, description="Why the check gave no verdict")
    produced_by: ProducedBy | None = None
