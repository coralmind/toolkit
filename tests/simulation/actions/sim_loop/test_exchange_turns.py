"""A turn number is one exchange: the user message and the assistant reply share it."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from burr.core import State
from coralmind_mock_run import build_mock_manifest

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.conversation_signal import ConversationSignal
from corallab.schemas.eval.run_case import RunCase
from corallab.simulation.actions.sim_loop.assemble_sim_result import AssembleSimResult
from corallab.simulation.actions.sim_loop.call_sut import CallSUT
from corallab.simulation.actions.sim_loop.generate_persona_message import (
    GeneratePersonaMessage,
)
from corallab.simulation.actions.sim_loop.send_opening import SendOpening
from corallab.simulation.setup.sut_runner import call_system_under_test_for_transcript
from tests._records import make_persona, make_test_case


async def test_the_opening_is_exchange_1() -> None:
    result = await SendOpening({"turns": ["hello"]}).run(
        State({"conversation_history": [], "turn_number": 0})
    )
    assert result["turn_number"] == 1
    assert result["conversation_history"][0].turn_number == 1
    assert result["conversation_history"][0].role == "user"


async def test_the_assistant_reply_keeps_the_user_exchange() -> None:
    action = CallSUT(MagicMock(), build_mock_manifest(), {})
    state = State(
        {
            "conversation_history": [Turn(turn_number=1, role="user", content="hello")],
            "turn_number": 1,
            "round_count": 0,
        }
    )
    with patch.object(action, "_call_with_retry", new=AsyncMock(return_value="hi there")):
        result = await action.run(state)
    reply = result["conversation_history"][-1]
    assert result["turn_number"] == 1
    assert result["round_count"] == 1
    assert reply.role == "assistant"
    assert reply.turn_number == 1


async def test_the_next_user_message_starts_the_next_exchange() -> None:
    generator = MagicMock()
    generator.generate_message = AsyncMock(return_value=("and then", ConversationSignal()))
    state = State(
        {
            "conversation_history": [
                Turn(turn_number=1, role="user", content="hello"),
                Turn(turn_number=1, role="assistant", content="hi there"),
            ],
            "turn_number": 1,
            "round_count": 1,
            "max_turns": 10,
        }
    )
    result = await GeneratePersonaMessage(generator).run(state)
    message = result["conversation_history"][-1]
    assert result["turn_number"] == 2
    assert message.role == "user"
    assert message.turn_number == 2


async def test_scripted_lines_share_a_turn_number_with_their_replies() -> None:
    case = RunCase(test_case=make_test_case(evaluation_level=2), persona=make_persona())
    replies = iter(["first reply", "second reply", "third reply"])

    async def complete(**_kwargs: object) -> MagicMock:
        response = MagicMock()
        response.text = next(replies)
        return response

    with patch("corallab.llm_client.llm_client.complete", new=complete):
        turns = await call_system_under_test_for_transcript(case, build_mock_manifest(), "e")

    assert [(turn.turn_number, turn.role) for turn in turns] == [
        (1, "user"),
        (1, "assistant"),
        (2, "user"),
        (2, "assistant"),
        (3, "user"),
        (3, "assistant"),
    ]


async def test_an_empty_reply_fails_at_that_exchange() -> None:
    case = RunCase(test_case=make_test_case(evaluation_level=2), persona=make_persona())
    replies = iter(["first reply", ""])

    async def complete(**_kwargs: object) -> MagicMock:
        response = MagicMock()
        response.text = next(replies)
        return response

    with patch("corallab.llm_client.llm_client.complete", new=complete):
        failure = await call_system_under_test_for_transcript(case, build_mock_manifest(), "e")

    assert failure.failed_at_turn == 2
    assert failure.reason == "empty_response"


async def test_total_turns_counts_exchanges() -> None:
    history = [
        Turn(turn_number=1, role="user", content="hello"),
        Turn(turn_number=1, role="assistant", content="hi there"),
        Turn(turn_number=2, role="user", content="and then"),
        Turn(turn_number=2, role="assistant", content="go on"),
    ]
    result = await AssembleSimResult(build_mock_manifest(), 3).run(
        State(
            {
                "test_case_id": "aimhef-0000f001",
                "conversation_history": history,
                "turn_number": 2,
                "exit_reason": "max_turns",
            }
        )
    )
    assert result["sim_result"].total_turns == 2
