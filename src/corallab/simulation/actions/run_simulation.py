"""RunSimulation action — runs the L3 simulation to generate a transcript."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import PerTestCaseState


class RunSimulation(Action):
    """Simulates one test case at L3 and writes its transcript record."""

    @property
    def reads(self) -> list[str]:
        return ["run_case", "manifest", "evaluation_id"]

    @property
    def writes(self) -> list[str]:
        # sim_result is None when the simulation failed. The fidelity check reads transcript.
        return ["transcript", "sim_result"]

    async def run(self, state: State[PerTestCaseState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.schemas.eval.transcript import SimFailure
        from corallab.simulation.setup import simulation_runner
        from corallab.simulation.setup.helpers import make_transcript

        case = state["run_case"]
        manifest = state["manifest"]
        outcome = await simulation_runner.run_simulation(case, manifest, state["evaluation_id"])
        if isinstance(outcome, SimFailure):
            failed = make_transcript(case, manifest, sim_failure=outcome)
            return {"transcript": failed, "sim_result": None}
        turns, sim_result = outcome
        return {
            "transcript": make_transcript(
                case, manifest, turns=turns, exit_reason=sim_result.exit_reason
            ),
            "sim_result": sim_result,
        }

    def update(self, result: dict, state: State[PerTestCaseState]) -> State[PerTestCaseState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
