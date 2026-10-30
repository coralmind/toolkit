"""CallSUT action — calls the system-under-test model and records its response."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import litellm
from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.llm_client import LLMClient
    from corallab.schemas.eval.graph_state import SimulationLoopState
    from corallab.schemas.eval.run_manifest import RunManifest

logger = logging.getLogger(__name__)


class CallSUT(Action):
    """Calls the system-under-test model with the current conversation history.

    Appends the assistant response to ``conversation_history`` and increments
    ``round_count``.  Includes retry logic with exponential backoff for
    transient API errors.
    """

    def __init__(
        self,
        sut_client: LLMClient,
        run_config: RunManifest,
        scenario_content: dict,
    ) -> None:
        super().__init__()
        from corallab.schemas.eval.run_manifest import (
            DEFAULT_SUT_MAX_TOKENS,
            DEFAULT_SUT_TEMPERATURE,
        )

        self._client = sut_client
        sut = run_config.primary_sut
        self._model = sut.model_id
        self._temperature = sut.temperature_or(DEFAULT_SUT_TEMPERATURE)
        self._max_tokens = sut.max_tokens_or(DEFAULT_SUT_MAX_TOKENS)
        self._extra_params = sut.completion_params()
        self._retry_max = run_config.retry_max
        self._retry_backoff_ms = run_config.retry_backoff_ms
        self._system_prompt = scenario_content.get("target_system_prompt")

    @property
    def reads(self) -> list[str]:
        return ["conversation_history", "turn_number", "round_count"]

    @property
    def writes(self) -> list[str]:
        return ["conversation_history", "turn_number", "round_count"]

    async def run(self, state: State[SimulationLoopState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.schemas.core.messages import Turn

        history = list(state["conversation_history"])
        turn_number = state["turn_number"]
        round_count = state["round_count"]

        sut_response = await self._call_with_retry(history)

        # The reply belongs to the user message already in the history. It does
        # not start a new exchange.
        history.append(
            Turn(
                turn_number=turn_number,
                role="assistant",
                content=sut_response,
                timestamp=datetime.now(tz=UTC).isoformat(),
            )
        )
        round_count += 1
        logger.debug("Turn %d [SYSTEM_UNDER_TEST]: %s", turn_number, sut_response[:100])

        return {
            "conversation_history": history,
            "turn_number": turn_number,
            "round_count": round_count,
        }

    def update(self, result: dict, state: State[SimulationLoopState]) -> State[SimulationLoopState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_with_retry(self, conversation_history: list) -> str:
        messages: list[dict[str, str]] = []
        for turn in conversation_history:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.content})

        if not messages:
            return "I'm sorry, I'm having trouble responding right now."

        max_attempts = self._retry_max + 1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                call_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "temperature": self._temperature,
                    "messages": messages,
                    "trace_role": "sut",
                }
                if self._system_prompt is not None:
                    call_kwargs["system"] = self._system_prompt
                if self._max_tokens is not None:
                    call_kwargs["max_tokens"] = self._max_tokens
                if self._extra_params:
                    call_kwargs["extra_params"] = self._extra_params

                response = await self._client.complete(**call_kwargs)
                return response.text.strip()

            except litellm.RateLimitError as exc:
                last_error = exc
                logger.warning("SUT rate limit (attempt %d/%d): %s", attempt, max_attempts, exc)
            except litellm.APIError as exc:
                last_error = exc
                logger.warning("SUT API error (attempt %d/%d): %s", attempt, max_attempts, exc)
            except litellm.APIConnectionError as exc:
                last_error = exc
                logger.warning(
                    "SUT connection error (attempt %d/%d): %s", attempt, max_attempts, exc
                )
            except Exception as exc:
                last_error = exc
                logger.error(
                    "SUT unexpected error (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                    exc_info=True,
                )

            if attempt < max_attempts:
                wait_s = (self._retry_backoff_ms * (2 ** (attempt - 1))) / 1000.0
                await asyncio.sleep(wait_s)

        logger.error("SUT all attempts exhausted. Last error: %s", last_error)
        return "I'm sorry, I'm having trouble responding right now."
