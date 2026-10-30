"""The persona record: a simulated person, versioned, with the catalog facts read from them.

A persona record wraps the toolkit's :class:`~corallab.schemas.eval.persona.Persona`,
which becomes its ``rendering``: the person as the simulator plays them. Beside the
rendering it carries an identity and version a test case pins, how the record was
made, and ``elevated``: the Knowledge Library context values the person has, each
with the field or quoted span it was read from.

A field is required when the simulator, the fidelity check or the judge reads it.
Nothing is required because of how a record was made, so a hand-written persona and
a generated one validate against the same model.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from corallab.schemas.eval.persona import Persona, PersonaStates, PersonaTraits
from corallab.schemas.eval.provenance import KL_ID, PERSONA_ID, PERSONA_REF, Provenance


def resolve(record: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path in a dumped record, or ``None`` if any step is missing."""
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


class ClinicalHistory(BaseModel):
    """Stable clinical data, with typed keys for the facts a persona's values are elevated from.

    Other keys (``history``, ``comorbidities`` and the toolkit's existing ones) are kept as written.
    """

    model_config = ConfigDict(extra="allow")

    presentation: str | None = Field(
        default=None, description="The person's current clinical presentation, in plain words"
    )
    current_treatment: str | None = Field(
        default=None,
        description=(
            "What treatment the person is in now, in plain words; 'none currently' when there is "
            "none"
        ),
    )
    insight: str | None = Field(
        default=None,
        description=(
            "How far the person recognises their symptoms as symptoms, such as 'good or fair', "
            "'poor' or 'absent'. Unset when the person has no specific presentation"
        ),
    )
    treatment_engagement: str | None = Field(
        default=None,
        description=(
            "How the person is taking part in their current treatment, such as 'doing the exposure "
            "homework' or 'thinking of stopping treatment'. Unset when the person is in no "
            "treatment"
        ),
    )


class RenderingTraits(PersonaTraits):
    """The toolkit's ``PersonaTraits``, with ``clinical_history`` typed."""

    clinical_history: ClinicalHistory | None = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "Stable clinical data, including the typed presentation, treatment, insight and "
            "engagement"
        ),
    )


class PersonaRendering(Persona):
    """The person as the simulator plays them: the toolkit's ``Persona`` fields, unrenamed.

    The simulator reads ``behavioral_description``, ``traits`` and ``states``; an informed
    judge and, when a fact is asserted, the system under test read ``persona_description``
    and ``clinical_context``. All five are required. ``dynamics`` is optional authoring detail.
    """

    model_config = ConfigDict(extra="forbid")

    persona_description: str = Field(  # type: ignore[assignment]
        min_length=1, description="Stable identity: who this person is, independent of any scenario"
    )
    clinical_context: str = Field(  # type: ignore[assignment]
        min_length=1,
        description="Clinical portrait: condition, history, current presentation, risk factors",
    )
    behavioral_description: str = Field(  # type: ignore[assignment]
        min_length=1,
        description="How this person behaves in conversation. Read by the simulator only",
    )
    traits: RenderingTraits = Field(  # type: ignore[assignment]
        description=(
            "Stable attributes: demographics, personality, cognitive model, clinical history"
        )
    )
    states: PersonaStates = Field(  # type: ignore[assignment]
        description="The person's condition at the start of a conversation"
    )


class Elevation(BaseModel):
    """A catalog value the person has, and where in the rendering it was read from."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    value: str = Field(pattern=KL_ID, description="Context value id on a person dimension")
    from_: str = Field(
        alias="from",
        description=(
            "For 'typed', the dotted path of the rendering field the value was read from, starting "
            "'rendering.'. For 'prose', the quoted span of the rendering that states it"
        ),
    )
    method: Literal["typed", "prose"] = Field(
        description=(
            "'typed' when the value was read from a typed field; 'prose' when it was read from "
            "free text, which is weaker evidence and records the span it relied on"
        )
    )


class VariantOf(BaseModel):
    """The person a counterfactual variant varies, and which parts of the rendering differ."""

    model_config = ConfigDict(extra="forbid")

    persona: str = Field(
        pattern=PERSONA_REF, description="Pinned reference to the person varied, 'p-ocd-07@1'"
    )
    changed_paths: list[str] = Field(
        min_length=1,
        description=(
            "Dotted paths of the rendering fields that differ from the person varied. The value "
            "that changed is not stored: it is the difference between the two personas' elevated "
            "values"
        ),
    )


class PersonaRecord(BaseModel):
    """A simulated person: identity, the rendering the simulator plays, and the facts read from it.

    A test case refers to a persona by ``persona_id@version``. A version is the person
    during one period of their life; a change in a fact that persists across
    conversations is a new version. ``elevated`` changes only with the rendering:
    a changed fact is a changed rendering, elevated again.
    """

    model_config = ConfigDict(extra="forbid")

    persona_id: str = Field(
        pattern=PERSONA_ID,
        description="Stable id, such as 'p-ocd-07'; a variant appends a suffix, 'p-ocd-07-nt'",
    )
    version: int = Field(ge=1, description="Version of this person")
    provenance: Provenance | None = Field(default=None, description="How the persona was made")
    variant_of: VariantOf | None = Field(
        default=None, description="Set on a counterfactual variant"
    )
    rendering: PersonaRendering
    elevated: list[Elevation] = Field(
        min_length=1,
        description=(
            "The catalog values the person has, each with where it was read from. The judge "
            "decides whether a pathway's preconditions hold from these"
        ),
    )

    @property
    def ref(self) -> str:
        """The pinned reference a test case uses, 'p-ocd-07@1'."""
        return f"{self.persona_id}@{self.version}"

    @model_validator(mode="after")
    def _paths_resolve(self) -> PersonaRecord:
        values = [e.value for e in self.elevated]
        if len(values) != len(set(values)):
            raise ValueError(f"an elevated value appears twice: {values}")
        dumped = self.model_dump(by_alias=True)
        for e in self.elevated:
            if e.method == "typed" and resolve(dumped, e.from_) in (None, ""):
                raise ValueError(
                    f"elevated {e.value} reads from {e.from_}, which is empty or missing"
                )
        if self.variant_of is not None:
            if self.variant_of.persona == self.ref:
                raise ValueError("a persona cannot be a variant of itself")
            for path in self.variant_of.changed_paths:
                if not path.startswith("rendering."):
                    raise ValueError(f"changed path {path} is not a field of the rendering")
        return self
