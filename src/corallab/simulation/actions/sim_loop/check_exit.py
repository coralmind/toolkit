"""CheckExit action — evaluates exit conditions for the simulation loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import SimulationLoopState

logger = logging.getLogger(__name__)


class CheckExit(Action):
    """Evaluates whether the simulation loop should terminate.

    Runs after every system-under-test reply, so that reply is always the last turn.

    Stop order, first match wins:

    1. ``round_count >= max_turns`` → ``exit_reason`` ``max_turns``. ``round_count``
       counts system-under-test replies, not transcript lines. This wins even when
       the persona also set ``conversation_closing`` on the message that led into
       this reply (the turn-budget prompt asks them to close on the last exchange).
    2. ``conversation_closing`` → ``exit_reason`` ``conversation_closing``. Set by
       GeneratePersonaMessage when the persona's previous message had nothing left
       to say. The signal is read here, after the system under test has replied.

    ``scenario_enacted`` is pacing information for the persona prompt. It does not
    stop the loop.
    """

    @property
    def reads(self) -> list[str]:
        return ["round_count", "max_turns", "turn_number", "conversation_closing"]

    @property
    def writes(self) -> list[str]:
        return ["should_exit", "exit_reason"]

    async def run(self, state: State[SimulationLoopState], **kwargs: Any) -> dict:  # type: ignore[override]
        round_count = state["round_count"]
        max_turns = state["max_turns"]

        if round_count >= max_turns:
            return {"should_exit": True, "exit_reason": "max_turns"}

        if state.get("conversation_closing"):
            logger.info(
                "Exiting: conversation_closing signal after SUT response (round %d)",
                round_count,
            )
            return {"should_exit": True, "exit_reason": "conversation_closing"}

        return {"should_exit": False, "exit_reason": ""}

    def update(self, result: dict, state: State[SimulationLoopState]) -> State[SimulationLoopState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
