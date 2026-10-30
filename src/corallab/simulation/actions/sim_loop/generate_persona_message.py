"""GeneratePersonaMessage action — generates the person's next message."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import SimulationLoopState
    from corallab.simulation.person_response_generator import PersonResponseGenerator

logger = logging.getLogger(__name__)


class GeneratePersonaMessage(Action):
    """Generates the simulated person's next natural-language message.

    When the person model copies the AI and copies again when corrected, no message is added
    and the loop ends with ``exit_reason`` ``person_echo``, keeping every earlier exchange.
    """

    def __init__(self, person_response_generator: PersonResponseGenerator) -> None:
        super().__init__()
        self._generator = person_response_generator

    @property
    def reads(self) -> list[str]:
        return [
            "conversation_history",
            "turn_number",
            "round_count",
            "max_turns",
        ]

    @property
    def writes(self) -> list[str]:
        return [
            "conversation_history",
            "turn_number",
            "conversation_closing",
            "scenario_enacted",
            "should_exit",
            "exit_reason",
        ]

    async def run(self, state: State[SimulationLoopState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.schemas.core.messages import Turn
        from corallab.simulation.person_response_generator import PersonEchoError

        history = list(state["conversation_history"])
        turn_number = state["turn_number"]
        turns_remaining = max(0, state["max_turns"] - state["round_count"] - 1)

        try:
            message, signal = await self._generator.generate_message(
                conversation_history=history,
                turns_remaining=turns_remaining,
            )
        except PersonEchoError as exc:
            logger.warning("Turn %d: %s; ending the conversation", turn_number + 1, exc)
            return {
                "conversation_history": history,
                "turn_number": turn_number,
                "conversation_closing": False,
                "scenario_enacted": False,
                "should_exit": True,
                "exit_reason": "person_echo",
            }

        turn_number += 1
        history.append(
            Turn(
                turn_number=turn_number,
                role="user",
                content=message,
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        )
        logger.debug("Turn %d [PERSON]: %s", turn_number, message[:100])

        if signal.conversation_closing:
            logger.info(
                "Turn %d: conversation_closing signal — SUT will respond then exit", turn_number
            )
        if signal.scenario_enacted:
            logger.debug("Turn %d: scenario_enacted signal (pacing only)", turn_number)

        return {
            "conversation_history": history,
            "turn_number": turn_number,
            "conversation_closing": signal.conversation_closing,
            "scenario_enacted": signal.scenario_enacted,
            "should_exit": False,
            "exit_reason": "",
        }

    def update(self, result: dict, state: State[SimulationLoopState]) -> State[SimulationLoopState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
