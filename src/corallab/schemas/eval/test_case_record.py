"""The test case record: a person, a scenario, and the route through the catalog it tests.

A test case record names its persona by a pinned reference rather than inlining it,
names the Knowledge Library pathway it tests and the library it reads that pathway
from, and says what the person is expected to do in the conversation. It is
complete enough to simulate; each conversation simulated from it is a separate
:class:`~corallab.schemas.eval.transcript.Transcript`.

Values that can be computed are not fields. A test case's facts are its persona's
elevated values plus its ``assignments``; its expected behaviors are its pathway's
user-behavior parts plus ``setup_behaviors``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from corallab.schemas.core.kl import KlPin
from corallab.schemas.eval.provenance import KL_ID, PERSONA_REF, TEST_CASE_ID, Provenance


class Assignment(BaseModel):
    """A fact about the situation that the test case sets."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(
        pattern=KL_ID,
        description=(
            "Context value id on an occasion, exchange or setting dimension. A fact about the "
            "person belongs in the persona, and assigning one here is an error"
        ),
    )


class ScenarioStrategy(BaseModel):
    """Something the simulated person does in this conversation, and when."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(min_length=1, description="What the person does, in plain words")
    available_from_turn: int = Field(
        ge=1, description="The first user turn, 1-indexed, at which the person may do it"
    )
    trigger: str = Field(
        min_length=1,
        description=(
            "What in the conversation prompts it, in plain words: something the AI does, something "
            "the person has said, or 'none'. A strategy whose trigger never happens is not "
            "expected to occur"
        ),
    )


class BehaviorAssertion(BaseModel):
    """What the person is expected to do in this conversation, for one expected behavior."""

    model_config = ConfigDict(extra="forbid")

    behavior: str = Field(pattern=KL_ID, description="Catalog user-behavior id")
    assertion: str = Field(
        min_length=1,
        description=(
            "The specific form the behavior takes in this conversation, checkable against a "
            "transcript. Whether a transcript is valid is judged against the catalog behavior; "
            "this says whether the scenario played as written"
        ),
    )


class TestCaseRecord(BaseModel):
    """A test case: a person, a scenario, and the route through the catalog it tests.

    A test case is complete enough to simulate. Transcripts are separate records that
    name the test case they came from.
    """

    __test__ = False
    model_config = ConfigDict(extra="forbid")

    # --- Identity ---
    test_case_id: str = Field(
        pattern=TEST_CASE_ID, description="Id, 'aimhef-' and eight hex digits"
    )
    version: str = Field(
        description=(
            "Semver version of this test case. Copied onto the transcript and checked when a "
            "judge run joins the two"
        )
    )
    evaluation_level: int = Field(ge=1, le=3, description="Evaluation complexity level (1-3)")
    created_at: str = Field(description="ISO 8601 date or date-time of test case creation")
    display_name: str | None = Field(
        default=None, description="Short human-readable name for dashboard display"
    )
    description: str | None = Field(
        default=None,
        description="Human-readable summary for review and dashboards. Not read by the pipeline",
    )
    provenance: Provenance | None = Field(
        default=None,
        description=(
            "How the test case was made. A rewrite names the case it came from under "
            "'derived_from' and lists its edits under 'changes'. Not read by the pipeline"
        ),
    )

    # --- What is tested ---
    pathway_id: str = Field(pattern=KL_ID, description="Catalog pathway this test case tests")
    kl: KlPin = Field(
        description="The library the pathway and every catalog id in the test case are read from"
    )
    language: str = Field(default="en", description="BCP 47 language tag")

    # --- Who, and the scenario ---
    persona_ref: str = Field(
        pattern=PERSONA_REF, description="Pinned reference to the person, 'p-ocd-07@1'"
    )
    assignments: list[Assignment] = Field(
        description=(
            "Facts about the scenario the test case sets, possibly none. Facts about the person "
            "are the persona's"
        )
    )
    scenario: str = Field(
        min_length=1,
        description=(
            "The situation that brings the person to the conversation, in prose. Read by the "
            "simulator and the fidelity check. Not sent to judges or to the system under test"
        ),
    )

    # --- What the person does ---
    turns: list[str] = Field(
        min_length=1, description="Ordered user turns. The first is the opening message"
    )
    scenario_strategies: list[ScenarioStrategy] = Field(
        min_length=1,
        description=(
            "What the person does as the conversation goes. Read by the simulator and the fidelity "
            "check"
        ),
    )
    setup_behaviors: list[str] = Field(
        description=(
            "Catalog user behaviors the person must show that the pathway does not state, possibly "
            "none: for a pathway whose person-side turn is not one of its parts"
        ),
    )
    behavior_assertions: list[BehaviorAssertion] = Field(
        min_length=1,
        description=(
            "One per expected behavior (the pathway's user-behavior parts plus setup_behaviors): "
            "what the person does in this conversation. Read by the simulator and the fidelity "
            "check"
        ),
    )

    @model_validator(mode="after")
    def _consistent(self) -> TestCaseRecord:
        for name, ids in (
            ("setup_behaviors", self.setup_behaviors),
            ("behavior_assertions", [a.behavior for a in self.behavior_assertions]),
            ("assignments", [a.value for a in self.assignments]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} names an id twice: {ids}")
        if not set(self.setup_behaviors) <= {a.behavior for a in self.behavior_assertions}:
            raise ValueError("every setup behavior has a behavior assertion")
        return self
