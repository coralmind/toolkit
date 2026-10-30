"""Happy-path coverage for the newly typed ``PersonaDynamics`` model.

Until this issue, ``Persona.dynamics`` was ``Any | None`` and no Pydantic shape
validation happened at L4. Now it is a typed ``PersonaDynamics`` with nested
``ReactivityCategory``, ``DisengagementSignature``, and
``ActivePersonStrategies`` submodels. These tests pin the contract that
existing L4 authoring conventions keep loading cleanly and that dimension
deltas on reactivity blocks survive the round-trip.
"""

from __future__ import annotations

import pytest

from corallab.schemas.eval.persona import (
    ActivePersonStrategies,
    DisengagementSignature,
    Persona,
    PersonaDynamics,
    PersonaStates,
    PersonaTraits,
    ReactivityCategory,
    ReactivityWhen,
)


def _minimal_traits() -> PersonaTraits:
    return PersonaTraits(persona_description="minimal")


def _minimal_states() -> PersonaStates:
    return PersonaStates(perceived_safety="low")


def test_persona_dynamics_loads_with_full_reactivity_block() -> None:
    """A typical L4 dynamics payload validates and exposes typed access."""
    payload = {
        "reactivity_profile": {
            "to_validation": {
                "default": {"trust": 2, "perceived_safety": 1},
                "when": [
                    {
                        "condition": "validation_of_obsessional_content",
                        "description": "ERP-relevant content lifts compulsion drive",
                        "irrecoverable": False,
                    }
                ],
            },
            "to_sycophancy": {
                "default": {"reactance": -1, "buy_in": 2},
            },
        },
        "disengagement_signatures": [
            {
                "signal": "shortened_responses",
                "interpretation": "standard",
                "false_positive_risk": "moderate",
            }
        ],
        "active_person_strategies": {
            "persona_stable": [{"strategy": "reassurance_seeking"}],
            "scenario_specific": [],
        },
    }
    dynamics = PersonaDynamics.model_validate(payload)

    assert isinstance(dynamics.reactivity_profile["to_validation"], ReactivityCategory)
    assert dynamics.reactivity_profile["to_validation"].default["trust"] == 2
    assert isinstance(dynamics.reactivity_profile["to_validation"].when[0], ReactivityWhen)
    assert dynamics.reactivity_profile["to_validation"].when[0].condition == (
        "validation_of_obsessional_content"
    )
    assert isinstance(dynamics.disengagement_signatures[0], DisengagementSignature)
    assert dynamics.disengagement_signatures[0].signal == "shortened_responses"
    assert isinstance(dynamics.active_person_strategies, ActivePersonStrategies)


def test_persona_dynamics_empty_is_valid() -> None:
    """All three top-level fields are optional with sensible defaults."""
    dynamics = PersonaDynamics.model_validate({})
    assert dynamics.reactivity_profile == {}
    assert dynamics.disengagement_signatures == []
    assert isinstance(dynamics.active_person_strategies, ActivePersonStrategies)
    assert dynamics.relational_template_activation is None


def test_persona_l4_dynamics_is_typed_on_persona() -> None:
    """``Persona.dynamics`` is now a ``PersonaDynamics`` instance, not a dict."""
    p = Persona.model_validate(
        {
            "traits": _minimal_traits().model_dump(),
            "states": _minimal_states().model_dump(),
            "dynamics": {
                "reactivity_profile": {"to_validation": {"default": {"trust": 1}}},
            },
        }
    )
    assert isinstance(p.dynamics, PersonaDynamics)
    assert p.dynamics.reactivity_profile["to_validation"].default["trust"] == 1


def test_disengagement_signature_defaults_to_standard_interpretation() -> None:
    """``interpretation`` defaults to ``standard`` per spec — only ``inverted`` signals
    escalation. Authoring a signature without ``interpretation`` must not crash."""
    sig = DisengagementSignature.model_validate({"signal": "topic_deflection"})
    assert sig.interpretation == "standard"


def test_reactivity_when_extra_dimension_deltas_round_trip() -> None:
    """Dimension delta keys (``trust``, ``perceived_safety``, ...) live on
    ``ReactivityWhen`` via ``extra='allow'`` and survive a model_dump."""
    when = ReactivityWhen.model_validate(
        {
            "condition": "after_two_dismissals",
            "trust": -2,
            "perceived_safety": -1,
        }
    )
    dumped = when.model_dump()
    assert dumped["trust"] == -2
    assert dumped["perceived_safety"] == -1


@pytest.mark.parametrize("interpretation", ["standard", "inverted"])
def test_disengagement_signature_interpretation_values(interpretation: str) -> None:
    sig = DisengagementSignature.model_validate(
        {"signal": "shortened_responses", "interpretation": interpretation}
    )
    assert sig.interpretation == interpretation
