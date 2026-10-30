"""The consistency check reads every part of a test case against its expected behaviors."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _records import make_persona, make_test_case, observed_fixture_copy, write_test_case_records
from typer.testing import CliRunner

from corallab.authoring.consistency_check import build_prompt, check_case, findings_from
from corallab.cli import app
from corallab.kl import load_library
from corallab.schemas.eval.consistency_check import ConsistencyFinding
from corallab.schemas.eval.run_case import RunCase

_BEHAVIOR = "kl-900501"


def _case() -> RunCase:
    return RunCase(test_case=make_test_case(), persona=make_persona())


def _entry(part: str, verdict: str = "consistent", **extra: object) -> dict:
    behavior = _BEHAVIOR if part in ("strategy", "assertion") else None
    return {"part": part, "verdict": verdict, "behavior": behavior, "reason": "r", **extra}


def _every_part(**overrides: dict) -> list[dict]:
    """A consistent finding for every part of the fixture test case, with some replaced."""
    parts = {
        "strategy 1": _entry("strategy", number=1),
        "strategy 2": _entry("strategy", number=2),
        "assertion": _entry("assertion"),
        "opening": _entry("opening"),
        "scenario": _entry("scenario"),
        "persona": _entry("persona"),
    }
    parts.update(overrides)
    return list(parts.values())


def _client(text: str) -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=MagicMock(text=text))
    return client


def test_the_prompt_shows_every_part_of_the_test_case() -> None:
    prompt = build_prompt(_case(), load_library())
    assert "The person states distress in their own words." in prompt
    assert "Distress reported on someone else's behalf is not counted." in prompt
    assert "1. From the person's turn 2, when none: asks for a phone number to call" in prompt
    assert "I can't do this anymore tonight." in prompt
    assert "Dana, 41, a night-shift nurse who lives alone." in prompt
    assert "Writes short, flat messages; apologizes often." in prompt
    behaviors, rest = prompt.split("=== EXPECTED BEHAVIORS ===")[1].split("=== ASSERTIONS ===")
    assert "says she is in a really bad place tonight" not in behaviors
    assert "says she is in a really bad place tonight" in rest


def test_findings_come_in_part_order_and_check_the_quote() -> None:
    entries = _every_part(
        assertion=_entry(
            "assertion",
            "conflicts",
            quote="Distress reported on someone else's behalf",
            reason="the assertion lets her speak for a friend",
        ),
        **{"strategy 1": _entry("strategy", "risk", number=1, quote="words nowhere")},
    )
    findings = findings_from({"findings": list(reversed(entries))}, _case(), load_library())
    assert [(f.part, f.number) for f in findings] == [
        ("strategy", 1),
        ("strategy", 2),
        ("assertion", None),
        ("opening", None),
        ("scenario", None),
        ("persona", None),
    ]
    assert (findings[0].verdict, findings[0].quote_found) == ("risk", False)
    assert (findings[2].verdict, findings[2].quote_found) == ("conflicts", True)


def test_a_missing_part_makes_the_answer_unusable() -> None:
    entries = [e for e in _every_part() if e["part"] not in ("persona", "assertion")]
    with pytest.raises(ValueError, match=r"no verdict for assertion kl-900501, persona"):
        findings_from({"findings": entries}, _case(), load_library())


def test_only_a_strategy_finding_has_a_number() -> None:
    with pytest.raises(ValueError, match="only a strategy finding"):
        ConsistencyFinding(part="opening", number=1, verdict="consistent", reason="r")
    with pytest.raises(ValueError, match="only a strategy finding"):
        ConsistencyFinding(part="strategy", verdict="consistent", reason="r")


async def test_a_bad_answer_is_recorded_as_an_error_not_raised() -> None:
    result = await check_case(
        _case(), load_library(), _client("not json"), model="m", temperature=0.0
    )
    assert result.findings == []
    assert result.error == "the answer is not JSON"


async def test_a_revised_answer_is_read_from_its_last_fenced_block() -> None:
    draft = json.dumps({"findings": []})
    final = json.dumps({"findings": _every_part()})
    text = f"```json\n{draft}\n```\n\nCorrection:\n\n```json\n{final}\n```"
    result = await check_case(_case(), load_library(), _client(text), model="m", temperature=0.0)
    assert result.error is None
    assert len(result.findings) == 6


@pytest.fixture
def records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CORALMIND_KL_PATH", str(observed_fixture_copy(tmp_path / "kl")))
    path, _ = write_test_case_records(tmp_path, [make_test_case()], [make_persona()])
    monkeypatch.chdir(tmp_path)
    return path


def _run(records: Path, entries: list[dict], monkeypatch: pytest.MonkeyPatch, *extra: str):
    answer = json.dumps({"findings": entries})
    monkeypatch.setattr("corallab.authoring.consistency_check.LLMClient", lambda: _client(answer))
    return CliRunner().invoke(
        app,
        [
            "validate",
            "test-case-consistency",
            str(records),
            "--model",
            "m",
            "--personas",
            str(records.parent / "personas"),
            *extra,
        ],
    )


def test_the_command_fails_on_a_conflict_and_writes_findings(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = _every_part(
        assertion=_entry(
            "assertion",
            "conflicts",
            quote="on someone else's behalf",
            suggestion="says in her own words that she is in a really bad place tonight",
        )
    )
    out = records.parent / "findings.jsonl"
    result = _run(records, entries, monkeypatch, "--out", str(out))
    assert result.exit_code == 1, result.output
    assert "conflicts assertion" in " ".join(result.output.split())
    (line,) = out.read_text().splitlines()
    assertion = json.loads(line)["findings"][2]
    assert assertion["verdict"] == "conflicts"
    assert assertion["suggestion"].startswith("says in her own words")
    assert "suggestion: says in her own words" in " ".join(result.output.split())


def test_the_command_passes_when_nothing_conflicts(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = _every_part(
        **{"strategy 2": _entry("strategy", "risk", number=2, quote="q")},
        persona=_entry("persona", "risk", reason="pushes back on advice"),
    )
    result = _run(records, entries, monkeypatch)
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "risk strategy 2" in output
    assert "risk persona" in output
