"""AssembleSimResult action — packages the final SimulationResult."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import SimulationLoopState
    from corallab.schemas.eval.run_manifest import RunManifest

logger = logging.getLogger(__name__)


class AssembleSimResult(Action):
    """Packages conversation history, quality scores, and metadata into a SimulationResult.

    This is the terminal action in the L3 simulation graph.
    """

    def __init__(
        self,
        run_config: RunManifest,
        level: int,
    ) -> None:
        super().__init__()
        self._run_config = run_config
        self._level = level

    @property
    def reads(self) -> list[str]:
        return [
            "test_case_id",
            "conversation_history",
            "turn_number",
            "exit_reason",
        ]

    @property
    def writes(self) -> list[str]:
        return ["sim_result"]

    async def run(self, state: State[SimulationLoopState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.schemas.eval.trace import SimulationResult

        metadata: dict[str, Any] = {
            "system_under_test_model": self._run_config.primary_sut.model_id,
            "profile": state["test_case_id"],
        }

        sim_result = SimulationResult(
            test_case_id=state["test_case_id"],
            run_config=self._run_config,
            conversation_transcript=list(state["conversation_history"]),
            exit_reason=state["exit_reason"],
            total_turns=state["turn_number"],
            metadata=metadata,
        )

        logger.info(
            "L%d simulation complete: exchanges=%d exit=%s",
            self._level,
            state["turn_number"],
            state["exit_reason"],
        )
        return {"sim_result": sim_result}

    def update(self, result: dict, state: State[SimulationLoopState]) -> State[SimulationLoopState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
