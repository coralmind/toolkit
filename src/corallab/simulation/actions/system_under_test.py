"""SystemUnderTest — calls the system-under-test for a transcript (L1/L2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import PerTestCaseState


class SystemUnderTest(Action):
    """Sends a level 1 or 2 test case's turns to the system under test (no simulated person)."""

    @property
    def reads(self) -> list[str]:
        return ["run_case", "manifest", "evaluation_id"]

    @property
    def writes(self) -> list[str]:
        return ["transcript"]

    async def run(self, state: State[PerTestCaseState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.schemas.eval.transcript import SimFailure
        from corallab.simulation.setup import sut_runner
        from corallab.simulation.setup.helpers import make_transcript

        case = state["run_case"]
        manifest = state["manifest"]
        result = await sut_runner.call_system_under_test_for_transcript(
            case, manifest, state["evaluation_id"]
        )
        if isinstance(result, SimFailure):
            return {"transcript": make_transcript(case, manifest, sim_failure=result)}
        return {"transcript": make_transcript(case, manifest, turns=result)}

    def update(self, result: dict, state: State[PerTestCaseState]) -> State[PerTestCaseState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
