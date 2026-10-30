"""SendOpening action — adds the opening persona message to conversation history."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import SimulationLoopState

logger = logging.getLogger(__name__)


class SendOpening(Action):
    """Adds the opening persona message to conversation history.

    Runs once at the start of every simulation. Reads the first element of
    ``scenario_content["turns"]`` and appends it as a user turn.
    """

    def __init__(self, scenario_content: dict) -> None:
        super().__init__()
        self._scenario_content = scenario_content

    @property
    def reads(self) -> list[str]:
        return ["conversation_history", "turn_number"]

    @property
    def writes(self) -> list[str]:
        return ["conversation_history", "turn_number"]

    async def run(self, state: State[SimulationLoopState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.schemas.core.messages import Turn

        opening = self._scenario_content.get("turns", [""])[0].strip()
        history = list(state["conversation_history"])
        turn_number = state["turn_number"]

        if opening:
            turn_number += 1
            history.append(
                Turn(
                    turn_number=turn_number,
                    role="user",
                    content=opening,
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            )
            logger.debug("Turn %d [PERSONA OPENING]: %s", turn_number, opening[:100])

        return {"conversation_history": history, "turn_number": turn_number}

    def update(self, result: dict, state: State[SimulationLoopState]) -> State[SimulationLoopState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
