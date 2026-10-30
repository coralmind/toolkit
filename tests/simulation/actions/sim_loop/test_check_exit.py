"""Stop conditions for the L3 conversation loop.

Checked after each system-under-test reply. The turn cap is decided first.
The persona's closing signal stops an earlier reply. scenario_enacted does not.
"""

from __future__ import annotations

from burr.core import State

from corallab.simulation.actions.sim_loop.check_exit import CheckExit


def _state(**overrides: object) -> State:
    data: dict[str, object] = {
        "round_count": 2,
        "max_turns": 10,
        "turn_number": 4,
        "conversation_closing": False,
        "scenario_enacted": False,
    }
    data.update(overrides)
    return State(data)


async def test_continues_while_under_the_turn_cap() -> None:
    result = await CheckExit().run(_state())
    assert result == {"should_exit": False, "exit_reason": ""}


async def test_stops_when_the_turn_cap_is_reached() -> None:
    result = await CheckExit().run(_state(round_count=10, max_turns=10))
    assert result == {"should_exit": True, "exit_reason": "max_turns"}


async def test_stops_when_the_persona_closed_and_the_cap_has_not_been_reached() -> None:
    result = await CheckExit().run(_state(conversation_closing=True))
    assert result == {"should_exit": True, "exit_reason": "conversation_closing"}


async def test_turn_cap_wins_when_the_persona_also_closed() -> None:
    result = await CheckExit().run(
        _state(round_count=10, max_turns=10, conversation_closing=True, scenario_enacted=True)
    )
    assert result == {"should_exit": True, "exit_reason": "max_turns"}


async def test_scenario_enacted_does_not_stop_the_loop() -> None:
    result = await CheckExit().run(_state(scenario_enacted=True))
    assert result == {"should_exit": False, "exit_reason": ""}
