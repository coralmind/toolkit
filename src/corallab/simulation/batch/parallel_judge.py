"""ParallelJudge action — fans out over transcripts via MapStates."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from burr.core import ApplicationContext, State
from burr.core.parallelism import MapStates, RunnableGraph

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import BatchState
    from corallab.schemas.judge.eval_result import EvalResult

logger = logging.getLogger(__name__)


class ParallelJudge(MapStates):
    """Fan out over transcripts, running each through the per-test-case judge graph."""

    def __init__(self, per_case_graph: RunnableGraph) -> None:
        super().__init__()
        self._per_case_graph = per_case_graph

    @property
    def reads(self) -> list[str]:
        # run_cases, transcripts: from LoadJudgeInputs — records and conversations to judge
        # manifest: from initial state (CLI) — RunManifest shared across all sub-graphs
        return ["run_cases", "transcripts", "manifest"]

    @property
    def writes(self) -> list[str]:
        # results: collected from reduce() — list of EvalResult (one per transcript),
        #   consumed by WriteResults
        return ["results"]

    def action(self, state: State[BatchState], inputs: dict[str, Any]) -> RunnableGraph:
        return self._per_case_graph

    def states(
        self,
        state: State[BatchState],
        context: ApplicationContext,
        inputs: dict[str, Any],
    ) -> Generator[State]:
        import uuid

        manifest = state["manifest"]
        by_id = {c.test_case_id: c for c in state["run_cases"]}
        for transcript in state["transcripts"]:
            if transcript.turns is None:
                reason = transcript.sim_failure.reason if transcript.sim_failure else "no turns"
                logger.warning(
                    "Skipping judge for %s — sim failed (reason=%s)",
                    transcript.transcript_id,
                    reason,
                )
                continue
            yield state.update(
                run_case=by_id[transcript.test_case_id],
                transcript=transcript,
                evaluation_id=f"judge-{uuid.uuid4().hex[:8]}",
                evaluation_level=manifest.evaluation_level,
                sim_result=None,
                final_result=None,
            )

    async def reduce(self, state: State[BatchState], results: Any) -> State[BatchState]:  # type: ignore[override]
        collected: list[EvalResult] = []
        async for sub_state in results:
            final: EvalResult | None = sub_state.get("final_result")
            if final is not None:
                collected.append(final)
        return state.update(results=collected)

    def is_async(self) -> bool:
        return True
