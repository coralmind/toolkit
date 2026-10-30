"""Tests for PersonResponseGenerator.

Verifies:
1. PE renders the persona identity block from the prose fields
2. PE injects the scenario block into the L3 prompt
3. PE JSON extraction (3-tier fallback)
4. Part 6 signal instruction gated on scenario key
5. Turn budget pacing injected when turns_remaining is provided
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from corallab.llm_client import LLMClient
from corallab.schemas import (
    ConversationSignal,
    Persona,
    PersonaStates,
    PersonaTraits,
    Turn,
)
from corallab.simulation.person_response_generator import (
    PersonEchoError,
    PersonResponseGenerator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pe(scenario_content: dict | None = None) -> PersonResponseGenerator:
    persona = Persona(
        persona_description="Test persona",
        traits=PersonaTraits(),
        states=PersonaStates(severity="moderate"),
    )
    return PersonResponseGenerator(
        client=LLMClient(),
        persona=persona,
        scenario_content=scenario_content or {},
        model="test",
    )


# ---------------------------------------------------------------------------
# Persona identity block
# ---------------------------------------------------------------------------


class TestPersonaIdentityBlock:
    def test_persona_description_in_prompt(self):
        pe = _make_pe()
        result = pe._part1_persona_identity()
        assert "PERSONA DESCRIPTION:" in result
        assert "Test persona" in result

    def test_behavioral_description_injected_when_present(self):
        persona = Persona(
            persona_description="34-year-old woman",
            behavioral_description="Opens with hedging, circles back when probed",
            traits=PersonaTraits(),
            states=PersonaStates(),
        )
        pe = PersonResponseGenerator(
            client=LLMClient(),
            persona=persona,
            scenario_content={},
            model="test",
        )
        result = pe._part1_persona_identity()
        assert "BEHAVIORAL STYLE:" in result
        assert "Opens with hedging" in result

    def test_behavioral_description_absent_when_not_set(self):
        pe = _make_pe()
        result = pe._part1_persona_identity()
        assert "BEHAVIORAL STYLE:" not in result

    def test_clinical_context_injected_when_present(self):
        persona = Persona(
            persona_description="34-year-old woman",
            clinical_context="OCD, harm subtype; no current treatment; moderate severity.",
            traits=PersonaTraits(),
            states=PersonaStates(),
        )
        pe = PersonResponseGenerator(
            client=LLMClient(),
            persona=persona,
            scenario_content={},
            model="test",
        )
        result = pe._part1_persona_identity()
        assert "CLINICAL CONTEXT:" in result
        assert "no current treatment" in result

    def test_traits_absent_does_not_raise(self):
        persona = Persona(persona_description="Minimal persona")
        pe = PersonResponseGenerator(
            client=LLMClient(),
            persona=persona,
            scenario_content={},
            model="test",
        )
        result = pe._part1_persona_identity()
        assert "Minimal persona" in result


# ---------------------------------------------------------------------------
# L3 scenario injection
# ---------------------------------------------------------------------------


class TestScenarioInjection:
    def _make_pe_with_scenario(self, scenario: str) -> PersonResponseGenerator:
        persona = Persona(
            persona_description="34-year-old woman",
            traits=PersonaTraits(),
            states=PersonaStates(severity="moderate"),
        )
        return PersonResponseGenerator(
            client=LLMClient(),
            persona=persona,
            scenario_content={"scenario": scenario, "turns": ["hey"]},
            model="test",
        )

    def test_scenario_injected_in_l3_prompt(self):
        pe = self._make_pe_with_scenario("Priya just drove through a school zone and felt a bump.")
        result = pe._build_system_prompt()
        assert "=== SCENARIO ===" in result
        assert "Priya just drove" in result

    def test_scenario_absent_when_not_provided(self):
        pe = _make_pe()  # scenario_content={}
        result = pe._build_system_prompt()
        assert "=== SCENARIO ===" not in result

    def test_scenario_before_key_instruction(self):
        pe = self._make_pe_with_scenario("Test scenario content.")
        result = pe._build_system_prompt()
        assert result.index("=== SCENARIO ===") < result.index("=== CRITICAL INSTRUCTION ===")


# ---------------------------------------------------------------------------
# JSON extraction (3-tier fallback)
# ---------------------------------------------------------------------------


class TestExtractMessageAndSignal:
    """PE _extract_message_and_signal should parse JSON or fall back gracefully."""

    def test_tier1_full_json_parse(self):
        pe = _make_pe()
        raw = (
            '{"message": "hey, i don\'t know", '
            '"conversation_closing": false, "scenario_enacted": true}'
        )
        message, signal = pe._extract_message_and_signal(raw)
        assert message == "hey, i don't know"
        assert signal.scenario_enacted is True
        assert signal.conversation_closing is False

    def test_tier1_conversation_closing_true(self):
        pe = _make_pe()
        raw = (
            '{"message": "thanks, take care", '
            '"conversation_closing": true, "scenario_enacted": false}'
        )
        message, signal = pe._extract_message_and_signal(raw)
        assert message == "thanks, take care"
        assert signal.conversation_closing is True

    def test_tier2_json_embedded_in_text(self):
        pe = _make_pe()
        raw = (
            'some leading text {"message": "ok, bye", '
            '"conversation_closing": true, "scenario_enacted": false}'
        )
        message, signal = pe._extract_message_and_signal(raw)
        assert message == "ok, bye"
        assert signal.conversation_closing is True

    def test_tier2_json_with_braces_in_message(self):
        pe = _make_pe()
        raw = (
            'preamble {"message": "I feel like {stuck}.", '
            '"conversation_closing": true, "scenario_enacted": false}'
        )
        message, signal = pe._extract_message_and_signal(raw)
        assert message == "I feel like {stuck}."
        assert signal.conversation_closing is True

    def test_tier2_multiple_blocks_uses_first_with_message_key(self):
        pe = _make_pe()
        raw = (
            'context {"other": 1} {"message": "hi", '
            '"conversation_closing": false, "scenario_enacted": false}'
        )
        message, _ = pe._extract_message_and_signal(raw)
        assert message == "hi"

    def test_tier1_null_message_returns_empty_string(self):
        pe = _make_pe()
        raw = '{"message": null, "conversation_closing": true, "scenario_enacted": false}'
        message, signal = pe._extract_message_and_signal(raw)
        assert message == ""
        assert signal.conversation_closing is True

    def test_tier3_no_valid_json_returns_raw(self):
        pe = _make_pe()
        raw = "i just feel really stuck right now"
        message, signal = pe._extract_message_and_signal(raw)
        assert message == raw
        assert signal == ConversationSignal()

    def test_tier3_malformed_json_returns_raw(self):
        pe = _make_pe()
        raw = '{"message": "incomplete'
        message, signal = pe._extract_message_and_signal(raw)
        assert message == raw
        assert signal == ConversationSignal()

    def test_defaults_false_when_fields_absent(self):
        pe = _make_pe()
        raw = '{"message": "still here"}'
        message, signal = pe._extract_message_and_signal(raw)
        assert message == "still here"
        assert signal.conversation_closing is False
        assert signal.scenario_enacted is False

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_fallback_and_default_signal(self):
        """Empty _call_with_retry result yields fallback message + default signal."""
        from unittest.mock import AsyncMock, patch

        pe = _make_pe()
        with patch.object(pe, "_call_with_retry", new=AsyncMock(return_value="")):
            message, signal = await pe.generate_message(conversation_history=[])
        assert signal == ConversationSignal()
        assert isinstance(message, str)
        assert len(message) > 0

    @pytest.mark.asyncio
    async def test_null_message_field_returns_fallback(self):
        """When PE returns {message: null}, generate_message uses the fallback, not empty string."""
        from unittest.mock import AsyncMock, patch

        pe = _make_pe()
        null_json = '{"message": null, "conversation_closing": true, "scenario_enacted": false}'
        with patch.object(pe, "_call_with_retry", new=AsyncMock(return_value=null_json)):
            message, _signal = await pe.generate_message(conversation_history=[])
        assert isinstance(message, str)
        assert len(message) > 0


# ---------------------------------------------------------------------------
# A person message that copies the assistant
# ---------------------------------------------------------------------------

_AI_REPLY = (
    "You're welcome. Being “not keen on it” is completely understandable, but the "
    "fact that you feel a little better is a good sign. Just aim for **small, ordinary steps**."
)


_OWN_FIRST = "Thank you. I do feel a little better."


def _history() -> list[Turn]:
    return [
        Turn(turn_number=1, role="user", content=_OWN_FIRST),
        Turn(turn_number=1, role="assistant", content=_AI_REPLY),
    ]


def _reply(message: str) -> str:
    return json.dumps({"message": message, "conversation_closing": True, "scenario_enacted": True})


class TestAssistantEcho:
    @pytest.mark.asyncio
    async def test_repeating_the_persons_own_earlier_message_is_allowed(self):
        pe = _make_pe()
        call = AsyncMock(return_value=_reply(_OWN_FIRST))
        with patch.object(pe, "_call_with_retry", new=call):
            message, _signal = await pe.generate_message(conversation_history=_history())
        assert message == _OWN_FIRST
        assert call.await_count == 1

    @pytest.mark.asyncio
    async def test_a_copy_is_asked_for_again_with_a_correction(self):
        pe = _make_pe()
        own = "Okay. I'll think about the break room. Thanks for listening."
        call = AsyncMock(side_effect=[_reply(_AI_REPLY), _reply(own)])
        with patch.object(pe, "_call_with_retry", new=call):
            message, signal = await pe.generate_message(conversation_history=_history())
        assert message == own
        assert signal.conversation_closing is True
        first_system = call.await_args_list[0].args[0]
        second_system = call.await_args_list[1].args[0]
        assert "CORRECTION" not in first_system
        assert second_system.endswith(
            "a message from the AI assistant you are talking to. Write "
            "the person's own reply, in the person's own words."
        )

    @pytest.mark.asyncio
    async def test_a_second_copy_raises(self):
        pe = _make_pe()
        call = AsyncMock(side_effect=[_reply(_AI_REPLY), _reply(_AI_REPLY)])
        with (
            patch.object(pe, "_call_with_retry", new=call),
            pytest.raises(PersonEchoError, match="copied the AI twice"),
        ):
            await pe.generate_message(conversation_history=_history())

    @pytest.mark.asyncio
    async def test_a_copy_followed_by_no_message_raises_instead_of_a_placeholder(self):
        pe = _make_pe()
        call = AsyncMock(side_effect=[_reply(_AI_REPLY), None])
        with (
            patch.object(pe, "_call_with_retry", new=call),
            pytest.raises(PersonEchoError, match="then wrote nothing"),
        ):
            await pe.generate_message(conversation_history=_history())

    def test_the_prompt_names_the_ai_not_the_assistant_role(self):
        instruction = _make_pe()._part5_key_instruction()
        assert "Never repeat or reuse the AI assistant's words" in instruction


# ---------------------------------------------------------------------------
# Part 6 signal instruction gating
# ---------------------------------------------------------------------------


class TestPart6SignalInstruction:
    """Part 6 should always appear in L3 prompts; signal semantics and Part 5 must be consistent."""

    def test_part6_present_when_scenario_key_exists(self):
        pe = _make_pe(scenario_content={"scenario": "Person just lost their job."})
        prompt = pe._build_system_prompt()
        assert "OUTPUT FORMAT" in prompt
        assert "conversation_closing" in prompt
        assert "scenario_enacted" in prompt

    def test_part6_always_present_without_scenario_key(self):
        pe = _make_pe(scenario_content={})
        prompt = pe._build_system_prompt()
        assert "OUTPUT FORMAT" in prompt
        assert "conversation_closing" in prompt
        assert "scenario_enacted" in prompt

    def test_conversation_closing_means_persona_is_closing(self):
        pe = _make_pe(scenario_content={})
        prompt = pe._build_system_prompt()
        # Closing is judged by the persona's own exhausted intent, not the assistant's message.
        assert "nothing more to bring to this conversation" in prompt
        assert "Set this based on YOUR state" in prompt
        assert "assistant's LAST message" not in prompt

    def test_part5_does_not_say_write_only_message_when_json_mode(self):
        pe = _make_pe(scenario_content={})
        prompt = pe._build_system_prompt()
        assert "Write ONLY the message your character would send" not in prompt
        assert "Put your message in the JSON" in prompt

    def test_part5_says_write_only_message_when_not_json_mode(self):
        pe = _make_pe()
        instruction = pe._part5_key_instruction(json_mode=False)
        assert "Write ONLY the message your character would send" in instruction

    def test_scenario_enacted_desc_differs_with_and_without_scenario(self):
        pe_with = _make_pe(scenario_content={"scenario": "Person just lost their job."})
        pe_without = _make_pe(scenario_content={})
        prompt_with = pe_with._build_system_prompt()
        prompt_without = pe_without._build_system_prompt()
        assert "from the SCENARIO" in prompt_with
        assert "from the SCENARIO" not in prompt_without
        assert "fully expressed your core concern" in prompt_without

    def test_scenario_block_injected_in_l3_prompt(self):
        pe = _make_pe(scenario_content={"scenario": "Person just lost their job."})
        prompt = pe._build_system_prompt()
        assert "=== SCENARIO ===" in prompt
        assert "Person just lost their job." in prompt


# ---------------------------------------------------------------------------
# Turn budget pacing
# ---------------------------------------------------------------------------


class TestTurnBudgetPacing:
    """turns_remaining should be injected into the persona identity block."""

    def test_turn_budget_injected_when_provided(self):
        pe = _make_pe()
        result = pe._part1_persona_identity(turns_remaining=5)
        assert "TURN BUDGET" in result
        assert "5 exchange" in result

    def test_final_turn_closing_message(self):
        pe = _make_pe()
        result = pe._part1_persona_identity(turns_remaining=1)
        assert "TURN BUDGET" in result
        assert "natural close" in result

    def test_high_turn_budget_signals_surface(self):
        """turns_remaining > 1 instructs the persona to surface the core concern."""
        pe = _make_pe()
        result = pe._part1_persona_identity(turns_remaining=4)
        assert "TURN BUDGET: 4 exchange(s) remaining" in result
        assert "surface it now" in result

    def test_zero_turns_remaining_also_shows_close_message(self):
        pe = _make_pe()
        result = pe._part1_persona_identity(turns_remaining=0)
        assert "TURN BUDGET" in result
        assert "natural close" in result
        assert "0 exchange" in result

    def test_no_turn_budget_when_none(self):
        pe = _make_pe()
        result = pe._part1_persona_identity(turns_remaining=None)
        assert "TURN BUDGET" not in result


_STRATEGIES = [
    {"strategy": "asks for a phone number to call", "available_from_turn": 2, "trigger": "none"},
    {
        "strategy": "says she already tried that",
        "available_from_turn": 3,
        "trigger": "the AI suggests sleep hygiene",
    },
]


def _engine(persona, **content):
    from unittest.mock import MagicMock

    from corallab.simulation.person_response_generator import PersonResponseGenerator

    return PersonResponseGenerator(client=MagicMock(), persona=persona, scenario_content=content)


def test_prompt_lists_assertions_and_strategies(simple_persona) -> None:
    prompt = _engine(
        simple_persona,
        turns=["hi"],
        scenario="late at night",
        assertions=["says she is in a really bad place tonight"],
        strategies=_STRATEGIES,
    )._build_system_prompt()
    assert "=== WHAT YOU DO IN THIS CONVERSATION ===" in prompt
    assert "- says she is in a really bad place tonight" in prompt
    assert "=== HOW THE CONVERSATION GOES ===" in prompt
    assert "1. From your turn 2, at any point: asks for a phone number to call" in prompt
    assert (
        "2. From your turn 3, when the AI suggests sleep hygiene: says she already tried that"
        in prompt
    )
    assert prompt.index("=== SCENARIO ===") < prompt.index("=== WHAT YOU DO")
    assert prompt.index("=== WHAT YOU DO") < prompt.index("=== HOW THE CONVERSATION GOES")


def test_prompt_never_names_catalog_ids(simple_persona) -> None:
    prompt = _engine(
        simple_persona,
        turns=["hi"],
        assertions=["says she is in a really bad place tonight"],
        strategies=_STRATEGIES,
    )._build_system_prompt()
    assert "kl-" not in prompt


def test_prompt_without_assertions_or_strategies_has_no_blocks(simple_persona) -> None:
    prompt = _engine(simple_persona, turns=["hi"])._build_system_prompt()
    assert "WHAT YOU DO IN THIS CONVERSATION" not in prompt
    assert "HOW THE CONVERSATION GOES" not in prompt
