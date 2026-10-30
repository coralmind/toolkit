"""The test-case consistency check: read every part of a test case against its expected behaviors.

One model call per test case, run once on the records, not per run. A strategy, assertion,
opening, scenario or persona that makes an expected behavior's criterion fail, or matches one of
its exclusions, makes the test fail by how it was written; the check names the part and the
catalog words it collides with.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from coralmap import KnowledgeLibrary
from pydantic import ValidationError

from corallab.fidelity.elicitation import parse_json_answer
from corallab.llm_client import LLMClient
from corallab.schemas.eval.consistency_check import ConsistencyCheckResult, ConsistencyFinding
from corallab.schemas.eval.run_case import RunCase

logger = logging.getLogger(__name__)

PROMPT_VERSION = "test-case-consistency-v1"
_MAX_TOKENS = 16000
_WHOLE_PARTS = ("opening", "scenario", "persona")


def _prompt_template(version: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{version}.txt").read_text(encoding="utf-8")


def behavior_text(behavior: str, library: KnowledgeLibrary) -> str:
    """The behavior as its catalog entry defines it: definition, window, criteria, exclusions."""
    entry = library.user_behaviors[behavior]
    body = entry.body
    criteria = body.observable_criteria or body.absence_criteria or ""
    return "\n".join(
        [
            f"- behavior: {behavior}",
            f"  label: {entry.label}",
            f"  definition: {entry.definition}",
            f"  detection_window: {body.detection_window.value}",
            f"  criteria: {criteria}",
            f"  exclusions: {body.exclusions}",
        ]
    )


def build_prompt(case: RunCase, library: KnowledgeLibrary, version: str = PROMPT_VERSION) -> str:
    record = case.test_case
    rendering = case.persona.rendering
    persona = "\n".join(
        f"{label}: {text}"
        for label, text in (
            ("Description", rendering.persona_description),
            ("Clinical context", rendering.clinical_context),
            ("Manner in conversation", rendering.behavioral_description),
        )
        if text
    )
    behaviors = "\n".join(behavior_text(a.behavior, library) for a in record.behavior_assertions)
    assertions = "\n".join(
        f"- behavior: {a.behavior}\n  assertion: {a.assertion}" for a in record.behavior_assertions
    )
    strategies = "\n".join(
        f"{n}. From the person's turn {s.available_from_turn}, when {s.trigger}: {s.strategy}"
        for n, s in enumerate(record.scenario_strategies, start=1)
    )
    return _prompt_template(version).format(
        scenario=record.scenario,
        opening=record.turns[0] if record.turns else "",
        persona=persona,
        behaviors=behaviors,
        assertions=assertions,
        strategies=strategies,
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"')).strip().lower()


def findings_from(
    parsed: dict[str, Any], case: RunCase, library: KnowledgeLibrary
) -> list[ConsistencyFinding]:
    """The model's answer as findings, one per part, in order. Raises ``ValueError`` when unusable.

    The order is the strategies by number, the assertions in the test case's order, then the
    opening, the scenario and the persona.
    """
    items = parsed.get("findings")
    if not isinstance(items, list):
        raise ValueError("no 'findings' list")
    record = case.test_case
    expected = {a.behavior for a in record.behavior_assertions}
    by_key: dict[tuple[str, int | str | None], ConsistencyFinding] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        part = item.get("part")
        behavior = item.get("behavior") or None
        quote = item.get("quote") or None
        found = None
        if quote and behavior in expected:
            found = _normalize(quote) in _normalize(behavior_text(behavior, library))
        number = item.get("number") if part == "strategy" else None
        try:
            finding = ConsistencyFinding(
                part=part,
                number=int(str(number)) if number is not None else None,
                verdict=item.get("verdict"),
                behavior=behavior if behavior in expected else None,
                quote=quote,
                quote_found=found,
                reason=str(item.get("reason") or ""),
                suggestion=item.get("suggestion") or None,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(f"unusable entry {item!r}: {exc}") from exc
        key: int | str | None = None
        if finding.part == "strategy":
            key = finding.number
        elif finding.part == "assertion":
            key = finding.behavior
        by_key[(finding.part, key)] = finding
    wanted: list[tuple[str, int | str | None]] = [
        ("strategy", n) for n in range(1, len(record.scenario_strategies) + 1)
    ]
    wanted += [("assertion", a.behavior) for a in record.behavior_assertions]
    wanted += [(part, None) for part in _WHOLE_PARTS]
    missing = [" ".join(str(x) for x in k if x is not None) for k in wanted if k not in by_key]
    if missing:
        raise ValueError(f"no verdict for {', '.join(missing)}")
    return [by_key[k] for k in wanted]


async def check_case(
    case: RunCase,
    library: KnowledgeLibrary,
    client: LLMClient,
    *,
    model: str,
    temperature: float | None,
    extra_params: dict[str, Any] | None = None,
) -> ConsistencyCheckResult:
    """Check one test case. A failed call or an unusable answer is recorded as ``error``."""
    record = case.test_case
    result = ConsistencyCheckResult(
        test_case_id=record.test_case_id,
        display_name=record.display_name or "",
        behaviors=[a.behavior for a in record.behavior_assertions],
        model=model,
        prompt_version=PROMPT_VERSION,
    )
    try:
        response = await client.complete(
            model=model,
            messages=[{"role": "user", "content": build_prompt(case, library)}],
            max_tokens=_MAX_TOKENS,
            temperature=temperature,
            trace_role="test_case_consistency",
            trace_test_case_id=record.test_case_id,
            extra_params=extra_params,
        )
        parsed = parse_json_answer(response.text or "")
        if parsed is None:
            raise ValueError("the answer is not JSON")
        result.findings = findings_from(parsed, case, library)
    except Exception as exc:  # one case's failure must not stop the others
        logger.warning("consistency check failed for %s: %s", record.test_case_id, exc)
        result.error = str(exc)[:500]
    return result


async def check_cases(
    cases: list[RunCase],
    library: KnowledgeLibrary,
    *,
    model: str,
    temperature: float | None,
    extra_params: dict[str, Any] | None = None,
    concurrency: int = 4,
    client: LLMClient | None = None,
) -> list[ConsistencyCheckResult]:
    """Check every test case, ``concurrency`` at a time, in file order."""
    client = client or LLMClient()
    gate = asyncio.Semaphore(concurrency)

    async def one(case: RunCase) -> ConsistencyCheckResult:
        async with gate:
            return await check_case(
                case,
                library,
                client,
                model=model,
                temperature=temperature,
                extra_params=extra_params,
            )

    return list(await asyncio.gather(*(one(c) for c in cases)))
