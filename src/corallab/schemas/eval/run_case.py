"""A test case record and the persona record it names, resolved together for a run."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.test_case_record import TestCaseRecord


class RunCase(BaseModel):
    """What the simulator and the judge read for one test case."""

    model_config = ConfigDict(frozen=True)

    test_case: TestCaseRecord
    persona: PersonaRecord

    @property
    def test_case_id(self) -> str:
        return self.test_case.test_case_id

    @property
    def facts(self) -> frozenset[str]:
        """The context value ids the test case records: elevated values plus assignments."""
        return frozenset(e.value for e in self.persona.elevated) | frozenset(
            a.value for a in self.test_case.assignments
        )

    @model_validator(mode="after")
    def _persona_is_the_one_named(self) -> RunCase:
        if self.persona.ref != self.test_case.persona_ref:
            raise ValueError(
                f"{self.test_case.test_case_id} names {self.test_case.persona_ref}, "
                f"not {self.persona.ref}"
            )
        return self
