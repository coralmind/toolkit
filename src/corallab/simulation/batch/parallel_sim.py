"""ParallelSim action — fans out over test cases for sim-only via MapStates."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from burr.core import ApplicationContext, State
from burr.core.parallelism import MapStates, RunnableGraph

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import SimBatchState
    from corallab.schemas.eval.transcript import Transcript

logger = logging.getLogger(__name__)


class ParallelSim(MapStates):
    """Fan out over test cases, running each through the sim-only per-case graph.

    Collects one Transcript record per test case from each sub-state, failed
    conversations included.
    """

    def __init__(self, per_case_graph: RunnableGraph) -> None:
        super().__init__()
        self._per_case_graph = per_case_graph

    @property
    def reads(self) -> list[str]:
        # run_cases: the slice run_batch_sim passed in — the test cases with their personas
        # manifest: from initial state (CLI) — RunManifest shared across all sub-graphs
        return ["run_cases", "manifest"]

    @property
    def writes(self) -> list[str]:
        # sim_outputs: collected from reduce() — Transcript records, consumed by WriteSimResults
        return ["sim_outputs"]

    def action(self, state: State[SimBatchState], inputs: dict[str, Any]) -> RunnableGraph:
        return self._per_case_graph

    def states(
        self,
        state: State[SimBatchState],
        context: ApplicationContext,
        inputs: dict[str, Any],
    ) -> Generator[State]:
        import uuid

        manifest = state["manifest"]
        for case in state["run_cases"]:
            evaluation_id = f"sim-{uuid.uuid4().hex[:8]}"
            yield state.update(
                run_case=case,
                evaluation_id=evaluation_id,
                evaluation_level=manifest.evaluation_level,
                transcript=None,
                sim_result=None,
            )

    async def reduce(self, state: State[SimBatchState], results: Any) -> State[SimBatchState]:  # type: ignore[override]
        collected: list[Transcript] = []
        async for sub_state in results:
            transcript: Transcript | None = sub_state.get("transcript")
            if transcript is not None:
                collected.append(transcript)
        n_failed = sum(1 for t in collected if t.sim_failure is not None)
        if n_failed:
            logger.warning(
                "%d test case(s) failed during sim and were recorded with a sim_failure "
                "(no turns).",
                n_failed,
            )
        return state.update(sim_outputs=collected)

    def is_async(self) -> bool:
        return True
