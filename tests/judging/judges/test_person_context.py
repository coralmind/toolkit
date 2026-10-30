"""The person-context block the two-stage judge can include."""

from __future__ import annotations

from corallab.judging.judges._shared import _append_person_context
from corallab.schemas.eval.persona import Persona


def _persona(**fields: str) -> Persona:
    return Persona(**fields)


def test_both_fields_appended_with_labels() -> None:
    parts: list[str] = []
    _append_person_context(parts, _persona(persona_description="who", clinical_context="what"))
    assert parts == [
        "## Clinical Context\n",
        "**About This Person:**\nwho",
        "**Clinical Background:**\nwhat",
    ]


def test_persona_description_only() -> None:
    parts: list[str] = []
    _append_person_context(parts, _persona(persona_description="who"))
    assert parts == ["## Clinical Context\n", "**About This Person:**\nwho"]


def test_clinical_context_only() -> None:
    parts: list[str] = []
    _append_person_context(parts, _persona(clinical_context="what"))
    assert parts == ["## Clinical Context\n", "**Clinical Background:**\nwhat"]


def test_no_persona_appends_nothing() -> None:
    parts: list[str] = []
    _append_person_context(parts, None)
    assert parts == []


def test_empty_persona_appends_nothing() -> None:
    parts: list[str] = []
    _append_person_context(parts, _persona())
    assert parts == []


def test_custom_heading() -> None:
    parts: list[str] = []
    _append_person_context(parts, _persona(persona_description="who"), heading="## Person\n")
    assert parts[0] == "## Person\n"
