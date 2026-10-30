"""FanOutJudges action — fans out over judge passes via MapStates."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from burr.core import ApplicationContext, State
from burr.core.parallelism import MapStates, RunnableGraph

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import PerTestCaseState
    from corallab.schemas.judge.eval_result import JudgePassInput, JudgePassResult

logger = logging.getLogger(__name__)


class FanOutJudges(MapStates):
    """Fan out over configured judge passes, running each through a single-judge sub-graph."""

    def __init__(self, single_judge_graph: RunnableGraph) -> None:
        super().__init__()
        self._single_judge_graph = single_judge_graph

    @property
    def reads(self) -> list[str]:
        # judge_pass_inputs: from PrepareJudgePasses — list of JudgePassInput
        #   to fan out over; each item becomes a separate sub-graph execution
        return ["judge_pass_inputs"]

    @property
    def writes(self) -> list[str]:
        # judge_pass_results: collected from reduce() — list of JudgePassResult
        #   from all sub-graph executions, consumed by MergeJudgeRuns
        return ["judge_pass_results"]

    def action(self, state: State[PerTestCaseState], inputs: dict[str, Any]) -> RunnableGraph:
        return self._single_judge_graph

    def states(
        self,
        state: State[PerTestCaseState],
        context: ApplicationContext,
        inputs: dict[str, Any],
    ) -> Generator[State]:
        judge_pass_inputs: list[JudgePassInput] = state["judge_pass_inputs"]
        for pass_input in judge_pass_inputs:
            yield state.update(
                judge_pass_input=pass_input,
                judge_pass_result=None,
            )

    async def reduce(  # type: ignore[override]
        self, state: State[PerTestCaseState], results: Any
    ) -> State[PerTestCaseState]:
        collected: list[JudgePassResult] = []
        async for sub_state in results:
            result: JudgePassResult | None = sub_state.get("judge_pass_result")
            if result is not None:
                collected.append(result)
        return state.update(judge_pass_results=collected)

    def is_async(self) -> bool:
        return True
