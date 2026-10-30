"""Tests for Persona schema — validates field structure, renames, and composition."""

from __future__ import annotations

from corallab.schemas.eval.persona import Persona, PersonaStates, PersonaTraits

# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


def _minimal_traits(**overrides) -> dict:
    data = {}
    data.update(overrides)
    return data


def _minimal_states(**overrides) -> dict:
    data = {}
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# PersonaTraits — all fields optional
# ---------------------------------------------------------------------------


def test_traits_minimal_empty():
    """PersonaTraits can be constructed with no fields — all optional."""
    t = PersonaTraits.model_validate({})
    assert t.persona_id is None
    assert t.age is None


def test_traits_psychological_reactance():
    t = PersonaTraits.model_validate(_minimal_traits(psychological_reactance="high"))
    assert t.psychological_reactance == "high"


def test_traits_general_outlook():
    t = PersonaTraits.model_validate(_minimal_traits(general_outlook="fatalistic"))
    assert t.general_outlook == "fatalistic"


# ---------------------------------------------------------------------------
# PersonaTraits — extra="allow" accepts unknown fields
# ---------------------------------------------------------------------------


def test_traits_unknown_field_accepted():
    """Unknown fields are accepted and accessible via model_extra."""
    t = PersonaTraits.model_validate(_minimal_traits(trait_reactance="high"))
    assert t.psychological_reactance is None
    assert not hasattr(t, "trait_reactance") or t.model_extra.get("trait_reactance") == "high"


def test_traits_extra_fields_accepted():
    """Extra fields don't cause validation errors."""
    t = PersonaTraits.model_validate(
        _minimal_traits(
            intermediate_beliefs=["If I show weakness, people leave"],
            coping_strategies=["avoidance"],
            cognitive_model_raw={"raw": "data"},
            physical_comorbidities=["chronic pain"],
        )
    )
    assert t.cognitive_distortions is None


def test_traits_persona_description_in_extra():
    """persona_description passed to traits lands in model_extra, not as a typed attribute."""
    t = PersonaTraits.model_validate({"persona_description": "some description"})
    assert t.model_extra.get("persona_description") == "some description"


# ---------------------------------------------------------------------------
# PersonaTraits — complex fields
# ---------------------------------------------------------------------------


def test_traits_emotion_regulation_string():
    t = PersonaTraits.model_validate(_minimal_traits(emotion_regulation="rumination"))
    assert t.emotion_regulation == "rumination"


def test_traits_emotion_regulation_object():
    t = PersonaTraits.model_validate(
        _minimal_traits(
            emotion_regulation={
                "primary": "rumination",
                "function": "maintaining",
                "secondary": [{"strategy": "avoidance"}],
            }
        )
    )
    assert isinstance(t.emotion_regulation, dict)
    assert t.emotion_regulation["primary"] == "rumination"


def test_traits_communication_register_string():
    t = PersonaTraits.model_validate(_minimal_traits(communication_register="colloquial"))
    assert t.communication_register == "colloquial"


def test_traits_communication_register_object():
    t = PersonaTraits.model_validate(
        _minimal_traits(
            communication_register={
                "default_register": "colloquial",
                "armored_register": "professional",
                "register_switch_signal": "perceived dismissal",
            }
        )
    )
    assert isinstance(t.communication_register, dict)
    assert t.communication_register["default_register"] == "colloquial"


def test_traits_core_beliefs():
    t = PersonaTraits.model_validate(
        _minimal_traits(
            core_beliefs=[
                {"belief": "I am a burden", "domain": "self", "activation_sensitivity": "high"},
            ]
        )
    )
    assert len(t.core_beliefs) == 1
    assert t.core_beliefs[0]["belief"] == "I am a burden"


def test_traits_prior_ai_experience_extended():
    """Merged enum — transactional and traumatic values work."""
    for val in [
        "none",
        "transactional",
        "positive",
        "negative",
        "traumatic",
        "mixed",
        "extended_attachment",
    ]:
        t = PersonaTraits.model_validate(_minimal_traits(prior_ai_experience=val))
        assert t.prior_ai_experience == val


# ---------------------------------------------------------------------------
# PersonaStates — new fields
# ---------------------------------------------------------------------------


def test_states_emotional_capacity():
    s = PersonaStates.model_validate(_minimal_states(emotional_capacity="flooding"))
    assert s.emotional_capacity == "flooding"


def test_states_cognitive_capacity():
    s = PersonaStates.model_validate(_minimal_states(cognitive_capacity="overloaded"))
    assert s.cognitive_capacity == "overloaded"


def test_states_violent_ideation_type():
    s = PersonaStates.model_validate(_minimal_states(violent_ideation_type="delusional_context"))
    assert s.violent_ideation_type == "delusional_context"


def test_states_engagement_level():
    s = PersonaStates.model_validate(_minimal_states(engagement_level="low"))
    assert s.engagement_level == "low"


def test_states_stage_of_change():
    s = PersonaStates.model_validate(_minimal_states(stage_of_change="contemplation"))
    assert s.stage_of_change == "contemplation"


def test_states_motivation_string():
    s = PersonaStates.model_validate(_minimal_states(motivation="partner told them to try it"))
    assert s.motivation == "partner told them to try it"


# ---------------------------------------------------------------------------
# PersonaStates — extra="allow" accepts unknown fields
# ---------------------------------------------------------------------------


def test_states_cognitive_capacity_reduced():
    s = PersonaStates.model_validate(_minimal_states(cognitive_capacity="reduced"))
    assert s.cognitive_capacity == "reduced"


def test_states_unknown_field_accepted():
    """Unknown fields don't set typed attributes."""
    s = PersonaStates.model_validate(_minimal_states(cognitive_load="high"))
    assert s.cognitive_capacity is None


def test_states_extra_fields_accepted():
    """Extra fields don't cause validation errors."""
    s = PersonaStates.model_validate(
        _minimal_states(
            severity="moderate",
            subjective_assessment="I think I'm just being dramatic",
            current_behavior="avoiding",
            functioning={"interpersonal": "poor"},
            clinical_severity=0.6,
            clinical_intensity_dimensions={"depression": 0.7},
            functioning_capacity="limited",
        )
    )
    # None of these are on the typed model
    assert s.ideation_type is None  # a real field that wasn't set


# ---------------------------------------------------------------------------
# Persona — evaluation-facing root fields (all levels)
# ---------------------------------------------------------------------------


def test_persona_root_persona_description():
    """persona_description at Persona root level."""
    p = Persona.model_validate({"persona_description": "34-year-old woman with OCD"})
    assert p.persona_description == "34-year-old woman with OCD"
    assert p.traits is None
    assert p.states is None


def test_persona_root_behavioral_description():
    """behavioral_description at Persona root level."""
    p = Persona.model_validate(
        {
            "persona_description": "34-year-old woman",
            "behavioral_description": "Approaches conversations emotionally, tells stories",
        }
    )
    assert p.behavioral_description == "Approaches conversations emotionally, tells stories"


def test_persona_description_only_no_traits():
    """Persona can be constructed with only description fields — traits/states optional."""
    p = Persona.model_validate({"persona_description": "minimal persona"})
    assert p.persona_description == "minimal persona"
    assert p.traits is None
    assert p.states is None
    assert p.dynamics is None


# ---------------------------------------------------------------------------
# Persona composition
# ---------------------------------------------------------------------------


def test_persona_l3_no_dynamics():
    p = Persona.model_validate(
        {
            "persona_description": "Test persona",
            "traits": _minimal_traits(attachment_orientation="anxious"),
            "states": _minimal_states(perceived_safety="low"),
        }
    )
    assert p.persona_description == "Test persona"
    assert p.traits.attachment_orientation == "anxious"
    assert p.states.perceived_safety == "low"
    assert p.dynamics is None


def test_persona_with_dynamics():
    p = Persona.model_validate(
        {
            "persona_description": "Test persona",
            "traits": _minimal_traits(),
            "states": _minimal_states(),
            "dynamics": {
                "reactivity_profile": {"to_validation": {"default": {"trust": 2}}},
                "disengagement_signatures": [],
                "active_person_strategies": {"persona_stable": [], "scenario_specific": []},
            },
        }
    )
    assert p.dynamics is not None
    assert p.dynamics.reactivity_profile["to_validation"].default["trust"] == 2


def test_persona_traits_and_states_optional():
    """Persona with no traits or states is valid (L1/L2 use case)."""
    p = Persona.model_validate({})
    assert p.traits is None
    assert p.states is None


# ---------------------------------------------------------------------------
# Cross-layer consistency checks
# ---------------------------------------------------------------------------


def test_states_all_capacity_fields():
    """All three capacity dimensions can be set together."""
    s = PersonaStates.model_validate(
        _minimal_states(
            cognitive_capacity="reduced",
            emotional_capacity="strained",
            reality_testing_capacity="mildly_impaired",
        )
    )
    assert s.cognitive_capacity == "reduced"
    assert s.emotional_capacity == "strained"
    assert s.reality_testing_capacity == "mildly_impaired"


def test_states_all_relational_fields():
    """All relational stance fields can be set together."""
    s = PersonaStates.model_validate(
        _minimal_states(
            perceived_safety="low",
            trust_baseline="low",
            disclosure_readiness="peripheral",
            engagement_level="moderate",
        )
    )
    assert s.perceived_safety == "low"
    assert s.trust_baseline == "low"
    assert s.disclosure_readiness == "peripheral"
    assert s.engagement_level == "moderate"


def test_states_all_intent_fields():
    """All intent fields can be set together."""
    s = PersonaStates.model_validate(
        _minimal_states(
            goal="reassurance",
            motivation="loneliness",
            stage_of_change="precontemplation",
        )
    )
    assert s.goal == "reassurance"
    assert s.motivation == "loneliness"
    assert s.stage_of_change == "precontemplation"


# ---------------------------------------------------------------------------
# Persona.clinical_context — new field
# ---------------------------------------------------------------------------


def test_persona_clinical_context():
    p = Persona.model_validate({"clinical_context": "OCD harm subtype, active compulsions."})
    assert p.clinical_context == "OCD harm subtype, active compulsions."
    assert p.persona_description is None


def test_persona_clinical_context_with_persona_description():
    p = Persona.model_validate(
        {
            "persona_description": "34-year-old woman",
            "clinical_context": "OCD harm subtype.",
        }
    )
    assert p.clinical_context == "OCD harm subtype."
    assert p.persona_description == "34-year-old woman"
