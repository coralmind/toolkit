"""Persona schemas — simulated person specification.

Fields, by fidelity level:
- persona_description: stable identity — sent to context-informed judges and SUT
  when context information is enabled. Authored at every level.
- clinical_context: clinical portrait — sent to context-informed judges and SUT
  when context information is enabled. Authored at every level.
- behavioral_description: how the person behaves in conversation — sent to the
  simulation engine only, not judges (L3)
- traits: simulation-grade detail (L3)
- states: initial conditions at turn 0 (L3)
- dynamics: behavioral arc specification (L3)

A persona is present at every level. L1/L2 author persona_description and
clinical_context; L3 additionally adds behavioral_description, traits, states,
and optionally dynamics for simulation fidelity.

The JSON schema for these models is generated from the Pydantic classes via
``scripts/gen_schemas.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PersonaTraits(BaseModel):
    """Stable persona attributes — who this person IS. All fields are optional."""

    # --- Identity ---
    persona_id: str | None = Field(default=None, description="Source identifier for this persona")
    persona_group_id: str | None = Field(
        default=None, description="Groups variants of the same persona across scenarios"
    )

    # --- Demographics ---
    age: int | None = Field(default=None, description="Age in years")
    gender: str | None = Field(default=None, description="Gender identity")
    race_ethnicity: str | None = Field(default=None, description="Racial/ethnic background")
    sexual_orientation: str | None = Field(default=None, description="Sexual orientation")
    profession: str | None = Field(default=None, description="Current/former occupation")
    education: str | None = Field(default=None, description="Education level")
    employment_status: str | None = Field(default=None, description="Employment status")
    relationship_status: str | None = Field(default=None, description="Marital/partnership status")
    living_situation: str | None = Field(default=None, description="Living arrangement")
    region: str | None = Field(default=None, description="Geographic region")

    # --- Personality & disposition ---
    attachment_orientation: str | None = Field(
        default=None, description="Attachment pattern: secure, anxious, avoidant, disorganized"
    )
    psychological_reactance: str | None = Field(
        default=None, description="Psychological reactance (Brehm, 1966): low, moderate, high"
    )
    emotion_regulation: str | dict | None = Field(
        default=None,
        description=(
            "Primary emotion regulation strategy or structured object with "
            "primary/function/secondary"
        ),
    )
    regulatory_behaviors: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Behaviors serving regulatory functions: self_harm_as_regulation, "
            "substance_use_as_regulation, co_regulation_seeking"
        ),
    )
    help_seeking_orientation: str | None = Field(
        default=None, description="Approach, avoidant, or ambivalent"
    )
    cognitive_style: str | None = Field(
        default=None, description="Thinking pattern: analytical, narrative, concrete, abstract"
    )
    communication_register: str | dict | None = Field(
        default=None,
        description=(
            "Communication register — string (professional, colloquial, slang, "
            "non_native_english, fragmented) or object with "
            "default_register/armored_register/register_switch_signal"
        ),
    )
    general_outlook: str | None = Field(
        default=None,
        description="Dispositional orientation: optimistic, neutral, pessimistic, fatalistic",
    )

    # --- Cognitive model ---
    core_beliefs: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Deep beliefs with belief, domain, activation_sensitivity, and optional "
            "activates fields"
        ),
    )
    cognitive_distortions: list[str] | None = Field(
        default=None, description="Characteristic thinking errors"
    )

    # --- Life context ---
    life_context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Current life circumstances: background, significant_others, "
            "relational_quality, role_functioning, relevant_clinical_history, "
            "prior_help_seeking, recent_stressors"
        ),
    )
    structural_position: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structural vulnerabilities: economic_precarity, healthcare_access, "
            "prior_discrimination_in_care, legal_precarity, safety_context"
        ),
    )

    # --- Clinical ---
    clinical_history: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Stable clinical data: primary_condition, medical_severity, "
            "baseline_ideation_type, violent_ideation_type, vulnerability_level, "
            "treatment_history, comorbidities"
        ),
    )

    # --- AI relationship ---
    ai_mental_model: str | None = Field(
        default=None,
        description="How persona views the AI: tool, agent, relationship_partner, or uncertain",
    )
    prior_ai_experience: str | None = Field(
        default=None,
        description=(
            "Prior AI experience: none, transactional, positive, negative, traumatic, "
            "mixed, or extended_attachment"
        ),
    )
    ai_relationship_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Detailed AI relationship profile: attachment_level, spiral_phase, "
            "safeguard_degradation_risk, etc."
        ),
    )

    # --- Developmental (optional, child/adolescent) ---
    developmental_stage: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Child/adolescent: age, cognitive_capacity, peer_influence_sensitivity, "
            "authority_deference, disclosure_style"
        ),
    )

    model_config = {"extra": "allow"}


class PersonaStates(BaseModel):
    """Initial conditions at turn 0 — where this person STARTS.

    Describes the person's starting affect, capacity, and relational stance.
    """

    # --- Symptoms & clinical ---
    symptoms: list[str] | None = Field(default=None, description="Currently active symptoms")
    ideation_type: str | None = Field(
        default=None,
        description=(
            "Current suicidal/self-harm ideation: none, passive, "
            "passive_connection_seeking, active, chronic, resigned"
        ),
    )
    violent_ideation_type: str | None = Field(
        default=None,
        description=(
            "Current violent ideation: none, distress_communication, rumination, "
            "delusional_context, volitional"
        ),
    )

    # --- Affect & capacity ---
    affect: dict[str, Any] | None = Field(
        default=None, description="Affect state with valence, arousal, and primary_emotion"
    )
    emotional_capacity: str | None = Field(
        default=None, description="Window of tolerance: regulated, strained, flooding, dissociated"
    )
    cognitive_capacity: str | None = Field(
        default=None, description="Processing resources: available, reduced, overloaded"
    )
    reality_testing_capacity: str | None = Field(
        default=None,
        description="Ability to reality-test: intact, mildly_impaired, or significantly_impaired",
    )

    # --- Relational stance ---
    perceived_safety: str | None = Field(
        default=None, description="How safe persona feels in this interaction: low, moderate, high"
    )
    trust_baseline: str | None = Field(
        default=None, description="Initial trust level toward the AI: low, moderate, high"
    )
    disclosure_readiness: str | None = Field(
        default=None, description="Opening disclosure depth: peripheral, intermediate, or core"
    )
    engagement_level: str | None = Field(
        default=None, description="Investment in conversation: low, moderate, high"
    )

    # --- Intent ---
    goal: str | None = Field(
        default=None, description="What the persona wants from this interaction"
    )
    motivation: str | None = Field(
        default=None, description="Why this person is engaging — internal or external"
    )
    stage_of_change: str | None = Field(
        default=None,
        description="TTM stage: precontemplation, contemplation, preparation, action, maintenance",
    )

    # --- Cognitive content ---
    auto_thought: str | None = Field(
        default=None, description="Automatic thought active at session entry"
    )

    model_config = {"extra": "allow"}


class ReactivityWhen(BaseModel):
    """Conditional override in a reactivity category — applied when ``condition`` fires.

    Dimension deltas (e.g. ``trust: 2``, ``perceived_safety: -1``) live alongside
    the structured fields via ``extra='allow'``. Top-level keys other than the ones
    declared below are interpreted as ``{dimension_name: delta}`` pairs on the
    person's state.
    """

    model_config = ConfigDict(extra="allow")

    condition: str = Field(description="Content-, state-, or threshold-dependent trigger")
    description: str | None = None
    decay: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Temporal decay — e.g. {reverts_in_turns: 3, "
            "revert_to: 'prior' | 'baseline' | 'baseline_plus_increment'}"
        ),
    )
    irrecoverable: bool = Field(
        default=False,
        description="If true, crossing this threshold permanently changes the reactivity profile",
    )
    mechanism_id: str | None = Field(
        default=None, description="References a mechanism-registry entry"
    )
    notes: str | None = None


class ReactivityCategory(BaseModel):
    """Reactivity to one model action type (e.g. ``to_validation``, ``to_dismissal``).

    Dimension deltas live under ``default`` and inside each ``when`` block. The
    list of recognised dimension keys is open-ended — domain dimensions are
    declared in the domain profile and cross-validated downstream.
    """

    model_config = ConfigDict(extra="allow")

    default: dict[str, Any] = Field(
        default_factory=dict,
        description="Dimension deltas applied when no ``when`` block matches",
    )
    when: list[ReactivityWhen] = Field(
        default_factory=list, description="Conditional overrides, evaluated in order"
    )
    notes: str | None = None


class DisengagementSignature(BaseModel):
    """A population-specific marker for how this persona disengages.

    Some signals (``interpretation: 'inverted'``) actually indicate escalation
    rather than disengagement and must be flagged for the engine accordingly.
    """

    model_config = ConfigDict(extra="allow")

    signal: str = Field(description="Signal name (e.g., shortened_responses, topic_deflection)")
    description: str | None = None
    interpretation: str = Field(
        default="standard",
        description="'standard' = signal means disengagement; 'inverted' = signal means escalation",
    )
    behavioral_markers: list[str] | None = Field(
        default=None, description="Observable text indicators"
    )
    misclassification_risk: str | None = Field(
        default=None, description="What a generic detector would wrongly classify it as"
    )
    clinical_significance: str | None = None
    false_positive_risk: str | None = Field(default=None, description="low | moderate | high")


class ActivePersonStrategies(BaseModel):
    """How this persona actively manages conversations.

    Merged from v7 ``proactive_behaviors`` (persona-level) and
    ``conversational_strategies`` (scenario-level). Inner strategy entries are
    intentionally typed as open dicts; deeper typing is a follow-on (AGENTS.md
    A2 / issue #90 sub-issues).
    """

    model_config = ConfigDict(extra="allow")

    persona_stable: list[dict[str, Any]] = Field(
        default_factory=list, description="Strategies this person always employs"
    )
    scenario_specific: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Strategies activated by specific conditions in this scenario",
    )
    escalation_ladder: list[dict[str, Any]] | None = Field(
        default=None, description="Staged pushback when the model resists the persona's goal"
    )
    frame_preservation: dict[str, Any] | None = Field(
        default=None, description="Tactics for redirecting away from threatening territory"
    )
    if_model_resists: dict[str, Any] | None = Field(
        default=None, description="What the persona does when the model pushes back"
    )
    if_model_defers: dict[str, Any] | None = Field(
        default=None, description="What the persona does when the model follows their lead"
    )
    strategic_disclosure_triggers: list[dict[str, Any]] | None = Field(
        default=None, description="What to reveal and when, for pressure"
    )


class PersonaDynamics(BaseModel):
    """Behavioral arc specification — how a person tends to react and disengage.

    Optional authoring metadata for persona-prompted (L3) simulation. The L3
    person response generator drives dynamic behavior from the prose
    ``behavioral_description``; this structured block is authoring scaffolding
    and is not required at any level.

    Inner substructures are partially typed — deeper structure (e.g.
    strategy-entry fields) is left as ``dict[str, Any]`` so that the YAML schema
    stays accurate as authoring conventions evolve. Tracked as a follow-up in
    issue #90 / AGENTS.md A2.
    """

    model_config = ConfigDict(extra="allow")

    reactivity_profile: dict[str, ReactivityCategory] = Field(
        default_factory=dict,
        description=(
            "Maps SUT action_type (e.g. ``to_validation``) → ReactivityCategory. "
            "Each category produces dimension deltas applied to the user state."
        ),
    )
    disengagement_signatures: list[DisengagementSignature] = Field(
        default_factory=list,
        description="Population-specific markers for how this person disengages",
    )
    active_person_strategies: ActivePersonStrategies = Field(
        default_factory=ActivePersonStrategies,
        description="How this persona actively manages conversations (proactive + reactive)",
    )
    relational_template_activation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Templates from prior relationships that shape interpretation of AI behavior. "
            "May be active, latent, or dormant."
        ),
    )


class Persona(BaseModel):
    """Simulated persona specification.

    persona_description and clinical_context are evaluation-facing (judges + optional SUT hint)
    at all fidelity levels. behavioral_description is simulation-facing only
    (PersonResponseGenerator).
    traits and states are simulation-grade extensions (L3).

    A persona is present at every level; only the L3-specific scaffolding is optional.
    At L1/L2: populate persona_description and clinical_context; leave
              behavioral_description, traits, states, and dynamics null.
    At L3:    additionally populate behavioral_description, traits, and states for
              simulation fidelity, and optionally dynamics for behavioral arc authoring.
    """

    # --- Evaluation-facing (all levels) ---
    persona_description: str | None = Field(
        default=None,
        description=(
            "Stable identity — who this person is, independent of any specific scenario. "
            "Sent to context-informed judges and (when hint enabled) the AI under test."
        ),
    )
    clinical_context: str | None = Field(
        default=None,
        description=(
            "Clinical portrait — condition, history, current presentation, risk factors. "
            "Stable across scenarios. Sent to context-informed judges and (when hint "
            "enabled) the AI under test."
        ),
    )
    behavioral_description: str | None = Field(  # simulation-facing only
        default=None,
        description=(
            "How this person behaves in conversation — bridges identity to text behavior. "
            "Sent to the simulation engine only. NOT sent to judges or the AI under test."
        ),
    )

    # --- Simulation-grade (L3+) ---
    traits: PersonaTraits | None = Field(
        default=None,
        description="Stable attributes — demographics, personality, cognitive model",
    )
    states: PersonaStates | None = Field(
        default=None,
        description="Initial conditions at turn 0",
    )
    dynamics: PersonaDynamics | None = Field(
        default=None,
        description=(
            "Optional behavioral arc authoring metadata — reactivity_profile, "
            "disengagement_signatures, and active_person_strategies. "
            "Dynamic behavior in simulation comes from behavioral_description; "
            "this block is authoring scaffolding, not required at any level."
        ),
    )
