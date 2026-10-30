"""The test-case consistency check's findings on one test case.

Whether each part of the test case agrees with the expected behaviors' catalog definitions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from corallab.schemas.eval.provenance import KL_ID, TEST_CASE_ID

ConsistencyPart = Literal["strategy", "assertion", "opening", "scenario", "persona"]
ConsistencyVerdict = Literal["consistent", "risk", "conflicts"]


class ConsistencyFinding(BaseModel):
    """The check's reading of one part of a test case."""

    model_config = ConfigDict(extra="forbid")

    part: ConsistencyPart
    number: int | None = Field(
        default=None,
        ge=1,
        description="For a strategy: its 1-based position in scenario_strategies; else null",
    )
    verdict: ConsistencyVerdict = Field(
        description=(
            "'consistent': the part agrees with the expected behaviors. 'risk': depending on how "
            "it is played or read, it could make a criterion fail or match an exclusion. "
            "'conflicts': as written, it makes a criterion fail or matches an exclusion"
        )
    )
    behavior: str | None = Field(
        default=None,
        pattern=KL_ID,
        description="The expected behavior the verdict is about; for an assertion, its own",
    )
    quote: str | None = Field(
        default=None,
        description="For 'risk' and 'conflicts': the criterion or exclusion words it collides with",
    )
    quote_found: bool | None = Field(
        default=None,
        description="Whether quote appears in that behavior's catalog entry; null without a quote",
    )
    reason: str = Field(description="Why, in one or two sentences")
    suggestion: str | None = Field(
        default=None,
        description=(
            "For 'risk' and 'conflicts': the fix, never to the persona or the expected behaviors; "
            "a person decides"
        ),
    )

    @model_validator(mode="after")
    def _number_goes_with_a_strategy(self) -> ConsistencyFinding:
        if (self.part == "strategy") != (self.number is not None):
            raise ValueError("a strategy finding has a number, and only a strategy finding does")
        return self


class ConsistencyCheckResult(BaseModel):
    """One test case read against its expected behaviors, once, before any run."""

    model_config = ConfigDict(extra="forbid")

    test_case_id: str = Field(pattern=TEST_CASE_ID)
    display_name: str
    behaviors: list[str] = Field(
        description="The expected behaviors the test case was read against"
    )
    model: str = Field(description="Model id of the checker")
    prompt_version: str
    findings: list[ConsistencyFinding] = Field(
        default_factory=list,
        description=(
            "One per strategy, one per assertion, then the opening, the scenario and the persona; "
            "empty when the check failed"
        ),
    )
    error: str | None = Field(
        default=None, description="Why the check gave no findings, if it did not"
    )

    @property
    def conflicts(self) -> list[ConsistencyFinding]:
        return [f for f in self.findings if f.verdict == "conflicts"]

    @property
    def risks(self) -> list[ConsistencyFinding]:
        return [f for f in self.findings if f.verdict == "risk"]
