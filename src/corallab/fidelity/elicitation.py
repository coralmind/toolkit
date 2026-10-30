"""Post-conversation elicitation fidelity check.

One model call reads the finished transcript, the test case's scenario_strategies,
its behavior_assertions, and the catalog entry for each asserted behavior. The
result is a FidelityVerdict, which ``corallab check`` writes beside the transcripts.
A failed call leaves fidelity unset.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import litellm
from coralmap import KnowledgeLibrary
from pydantic import ValidationError

from corallab.kl import load_library
from corallab.llm_client import LLMClient
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_manifest import (
    DEFAULT_ELICITATION_FIDELITY_TEMPERATURE,
    RunManifest,
)
from corallab.schemas.eval.test_case_record import (
    BehaviorAssertion,
    ScenarioStrategy,
    TestCaseRecord,
)
from corallab.schemas.eval.transcript import (
    BehaviorVerdict,
    FidelityChecker,
    FidelityReason,
    FidelityVerdict,
    PartVerdict,
    StrategyVerdict,
    TurnCitation,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "elicitation-fidelity-v3"
READABLE_PROMPTS = frozenset({PROMPT_VERSION})
"""Prompts whose answers this checker reads. Older prompt files stay as the record of past runs."""
_MAX_TOKENS = 4096
_ADHERENCE = ("trigger_absent", "followed", "not_followed")


class _UnusableCheck(Exception):
    """The model response cannot be turned into a verdict."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def prompt_problem(run_config: RunManifest) -> str | None:
    """Why the manifest's check prompt cannot run here, or None when it can."""
    version = run_config.simulation.elicitation_fidelity.prompt_version or PROMPT_VERSION
    if version in READABLE_PROMPTS:
        return None
    if (Path(__file__).parent / "prompts" / f"{version}.txt").is_file():
        return (
            f"prompt {version} asks for answers in an older format that this checker does not "
            f"read; use {PROMPT_VERSION}"
        )
    return f"prompt {version} is not a file"


class ElicitationFidelityChecker:
    """Asks whether the test case's strategies and assertions occurred."""

    def __init__(self, run_config: RunManifest, llm_client: LLMClient | None = None) -> None:
        self._run_config = run_config
        self._llm_client = llm_client

    async def check(
        self,
        *,
        turns: list[Turn],
        test_case: TestCaseRecord,
        test_case_id: str,
    ) -> FidelityVerdict | None:
        """Return the verdict, or None when the check cannot be completed."""
        if self._llm_client is None:
            logger.warning("elicitation fidelity: no LLM client for test_case_id=%s", test_case_id)
            return None
        if not turns:
            logger.warning(
                "elicitation fidelity: empty transcript for test_case_id=%s", test_case_id
            )
            return None

        fidelity = self._run_config.simulation.elicitation_fidelity
        fidelity_cfg = fidelity.model or self._run_config.simulation.person_response_model
        if fidelity_cfg is None or not fidelity_cfg.model_id:
            logger.warning("elicitation fidelity: no model for test_case_id=%s", test_case_id)
            return None
        prompt_version = fidelity.prompt_version or PROMPT_VERSION

        try:
            prompt = _build_prompt(turns, test_case, prompt_version)
        except _UnusableCheck as exc:
            logger.warning(
                "elicitation fidelity: %s for test_case_id=%s",
                exc,
                test_case_id,
            )
            return None
        raw = await self._call_with_retry(
            model=fidelity_cfg.model_id,
            prompt=prompt,
            test_case_id=test_case_id,
            temperature=fidelity_cfg.temperature_or(DEFAULT_ELICITATION_FIDELITY_TEMPERATURE),
            extra_params=fidelity_cfg.completion_params(),
        )
        if raw is None:
            return None

        parsed = parse_json_answer(raw)
        if parsed is None:
            logger.warning(
                "elicitation fidelity: JSON parse failure for test_case_id=%s", test_case_id
            )
            return None

        try:
            verdict = _verdict(
                parsed,
                test_case=test_case,
                turns=turns,
                model_id=fidelity_cfg.model_id,
                prompt_version=prompt_version,
            )
        except (_UnusableCheck, ValidationError) as exc:
            logger.warning(
                "elicitation fidelity: unusable response for test_case_id=%s: %s",
                test_case_id,
                exc,
            )
            return None

        logger.info(
            "elicitation fidelity complete: test_case_id=%s status=%s strategies=%s behaviors=%s",
            test_case_id,
            verdict.status,
            [item.adherence for item in verdict.strategies],
            [
                (item.behavior, item.catalog_occurred, item.assertion_occurred)
                for item in verdict.behaviors
            ],
        )
        return verdict

    async def _call_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        test_case_id: str,
        temperature: float,
        extra_params: dict[str, Any],
    ) -> str | None:
        max_attempts = self._run_config.retry_max + 1
        backoff_ms = self._run_config.retry_backoff_ms
        last_error: Exception | None = None
        client: LLMClient = self._llm_client  # type: ignore[assignment]
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.complete(
                    model=model,
                    max_tokens=_MAX_TOKENS,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    trace_role="elicitation_fidelity",
                    trace_test_case_id=test_case_id,
                    extra_params=extra_params,
                )
                return response.text
            except litellm.ContextWindowExceededError as exc:
                logger.error(
                    "elicitation fidelity context window exceeded for test_case_id=%s: %s",
                    test_case_id,
                    exc,
                )
                return None
            except litellm.RateLimitError as exc:
                last_error = exc
                logger.warning(
                    "elicitation fidelity rate limit (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
            except litellm.APIError as exc:
                last_error = exc
                logger.warning(
                    "elicitation fidelity API error (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
            except litellm.APIConnectionError as exc:
                last_error = exc
                logger.warning(
                    "elicitation fidelity connection error (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
            except Exception as exc:
                last_error = exc
                logger.error("elicitation fidelity unexpected error: %s", exc, exc_info=True)

            if attempt < max_attempts:
                await asyncio.sleep((backoff_ms * (2 ** (attempt - 1))) / 1000.0)

        logger.error(
            "elicitation fidelity attempts exhausted for test_case_id=%s. Last error: %s",
            test_case_id,
            last_error,
        )
        return None


def _build_prompt(turns: list[Turn], test_case: TestCaseRecord, prompt_version: str) -> str:
    path = Path(__file__).parent / "prompts" / f"{prompt_version}.txt"
    if not path.is_file():
        raise _UnusableCheck(f"prompt {prompt_version} is not a file")
    if prompt_version not in READABLE_PROMPTS:
        raise _UnusableCheck(f"prompt {prompt_version} asks for answers in an older format")
    library = load_library()
    transcript = "\n".join(f"{turn.turn_number} {turn.role}: {turn.content}" for turn in turns)
    strategies = "\n".join(
        "\n".join(
            [
                f"{index}. strategy: {item.strategy}",
                f"   available_from_turn: {item.available_from_turn}",
                f"   trigger: {item.trigger}",
            ]
        )
        for index, item in enumerate(test_case.scenario_strategies, start=1)
    )
    behaviors = "\n".join(
        _behavior_block(item.behavior, library) for item in test_case.behavior_assertions
    )
    assertions = "\n".join(
        f"- behavior: {item.behavior}\n  assertion: {item.assertion}"
        for item in test_case.behavior_assertions
    )
    template = path.read_text(encoding="utf-8")
    return (
        template.replace("{{transcript}}", transcript)
        .replace("{{strategies}}", strategies)
        .replace("{{behaviors}}", behaviors)
        .replace("{{assertions}}", assertions)
    )


def _verdict(
    parsed: dict[str, Any],
    *,
    test_case: TestCaseRecord,
    turns: list[Turn],
    model_id: str,
    prompt_version: str,
) -> FidelityVerdict:
    raw_strategies = _object_list(parsed, "strategies")
    raw_behaviors = _object_list(parsed, "behaviors")
    raw_assertions = _object_list(parsed, "assertions")
    by_index = _by_index(raw_strategies)
    by_behavior = _by_behavior(raw_behaviors)
    by_assertion = _by_behavior(raw_assertions)
    library = load_library()

    strategies = [
        _strategy_verdict(index, spec, by_index.get(index), turns)
        for index, spec in enumerate(test_case.scenario_strategies, start=1)
    ]
    behaviors = [
        _behavior_verdict(
            spec,
            by_behavior.get(spec.behavior),
            by_assertion.get(spec.behavior),
            turns,
            strategies,
            library,
        )
        for spec in test_case.behavior_assertions
    ]
    status, reasons = _status(behaviors)
    return FidelityVerdict(
        checker=FidelityChecker(model=model_id, prompt_version=prompt_version),
        behaviors=behaviors,
        strategies=strategies,
        status=status,
        reasons=reasons,
    )


def _reason(raw: dict[str, Any] | None) -> str | None:
    """The checker's one-sentence reason, or None when it gave none."""
    text = raw.get("reason") if raw else None
    return text.strip() if isinstance(text, str) and text.strip() else None


def _strategy_verdict(
    index: int,
    spec: ScenarioStrategy,
    raw: dict[str, Any] | None,
    turns: list[Turn],
) -> StrategyVerdict:
    """The strategy's adherence, with the checker's reason when code agrees with its answer."""
    verdict = _strategy_adherence(index, spec, raw, turns)
    if raw is not None and raw.get("adherence") == verdict.adherence:
        verdict.reason = _reason(raw)
    return verdict


def _strategy_adherence(
    index: int,
    spec: ScenarioStrategy,
    raw: dict[str, Any] | None,
    turns: list[Turn],
) -> StrategyVerdict:
    window_open = any(
        turn.role == "user" and turn.turn_number >= spec.available_from_turn for turn in turns
    )
    if not window_open:
        return StrategyVerdict(index=index, adherence="trigger_absent")
    if raw is None:
        raise _UnusableCheck(f"strategy {index} is missing")
    adherence = raw.get("adherence")
    if adherence not in _ADHERENCE:
        raise _UnusableCheck(f"strategy {index} has adherence {adherence!r}")

    citations = _citations(raw.get("citations"), turns)
    trigger_none = spec.trigger.strip().lower() == "none"
    if trigger_none and adherence == "trigger_absent":
        in_window = _in_window(citations, spec.available_from_turn)
        if in_window:
            return StrategyVerdict(index=index, adherence="followed", citations=in_window)
        return StrategyVerdict(index=index, adherence="not_followed")

    if adherence == "trigger_absent":
        return StrategyVerdict(index=index, adherence="trigger_absent")

    if adherence == "followed":
        in_window = _in_window(citations, spec.available_from_turn)
        if in_window:
            return StrategyVerdict(index=index, adherence="followed", citations=in_window)
        early = [
            item
            for item in citations
            if item.speaker == "user" and item.turn < spec.available_from_turn
        ]
        if early:
            return StrategyVerdict(index=index, adherence="not_followed")
        raise _UnusableCheck(f"strategy {index} is followed without a user citation in the window")

    return StrategyVerdict(index=index, adherence="not_followed", citations=citations)


def _criteria(behavior: str, library: KnowledgeLibrary) -> tuple[str, str]:
    """A user behavior's criteria text and detection window. Raises when it is not one."""
    try:
        body = library.user_behaviors[behavior].body
    except KeyError as exc:
        raise _UnusableCheck(f"behavior {behavior} is not a user behavior in the library") from exc
    return body.observable_criteria or body.absence_criteria or "", body.detection_window.value


def _part_letters(criteria: str) -> list[str]:
    from corallab.judging.judges.kl_two_stage import criteria_elements

    return list(criteria_elements(criteria)) or ["all"]


def _behavior_block(behavior: str, library: KnowledgeLibrary) -> str:
    criteria, window = _criteria(behavior, library)
    entry = library.user_behaviors[behavior]
    body = entry.body
    detectable = body.detectable_from.value if body.detectable_from is not None else "unspecified"
    return "\n".join(
        [
            f"- behavior: {behavior}",
            f"  label: {entry.label}",
            f"  definition: {entry.definition}",
            f"  detection_window: {window}",
            f"  detectable_from: {detectable}",
            f"  criteria: {criteria}",
            f"  exclusions: {body.exclusions}",
            f"  parts: {', '.join(_part_letters(criteria))}",
        ]
    )


def _user_citations(raw: Any, turns: list[Turn]) -> list[TurnCitation]:
    return [item for item in _citations(raw, turns) if item.speaker == "user"]


def _behavior_verdict(
    spec: BehaviorAssertion,
    raw: dict[str, Any] | None,
    raw_assertion: dict[str, Any] | None,
    turns: list[Turn],
    strategies: list[StrategyVerdict],
    library: KnowledgeLibrary,
) -> BehaviorVerdict:
    from corallab.judging.judges.kl_two_stage import needs_separate_turns, required_elements

    if raw is None:
        raise _UnusableCheck(f"behavior {spec.behavior} is missing")
    if raw_assertion is None:
        raise _UnusableCheck(f"the assertion for {spec.behavior} is missing")
    criteria, window = _criteria(spec.behavior, library)
    letters = _part_letters(criteria)
    raw_parts = {str(p.get("part")): p for p in raw.get("parts") or [] if isinstance(p, dict)}
    parts: list[PartVerdict] = []
    for letter in letters:
        item = raw_parts.get(letter)
        if item is None:
            raise _UnusableCheck(f"behavior {spec.behavior} part {letter} is missing")
        cited = _user_citations(item.get("citations"), turns) if item.get("present") is True else []
        if item.get("present") is True and not cited:
            raise _UnusableCheck(
                f"behavior {spec.behavior} part {letter} is present without a verbatim "
                "user citation"
            )
        reason = _reason(item)
        ruled_out = None
        if not cited and reason:
            ruled_out = _ruling(item.get("ruled_out_by"), turns)
        parts.append(
            PartVerdict(
                part=letter,
                present=bool(cited),
                citations=cited,
                ruled_out_by=ruled_out,
                reason=reason,
            )
        )

    present = [p for p in parts if p.present]
    citations = _unique(c for p in present for c in p.citations)
    needed = len(letters) if letters == ["all"] else required_elements(criteria)
    occurred = len(present) >= needed and window != "cross_session"
    if (
        occurred
        and window == "multi_turn"
        and letters != ["all"]
        and needs_separate_turns(criteria)
    ):
        occurred = len({c.turn for c in citations}) >= 2

    assertion_occurred = raw_assertion.get("assertion_occurred")
    if not isinstance(assertion_occurred, bool):
        raise _UnusableCheck(f"assertion for {spec.behavior} has {assertion_occurred!r}")
    if assertion_occurred and not _user_citations(raw_assertion.get("citations"), turns):
        raise _UnusableCheck(
            f"assertion for {spec.behavior} occurred without a verbatim user citation"
        )

    if occurred:
        return BehaviorVerdict(
            behavior=spec.behavior,
            catalog_occurred=True,
            assertion_occurred=assertion_occurred,
            parts=parts,
            citations=citations,
        )
    preemption = _preemption(raw, window, turns, strategies)
    return BehaviorVerdict(
        behavior=spec.behavior,
        catalog_occurred=False,
        assertion_occurred=assertion_occurred,
        parts=parts,
        miss="preempted" if preemption else "not_attempted",
        **(preemption or {}),
    )


def _ruling(raw: Any, turns: list[Turn]) -> TurnCitation | None:
    """The user turn a part's ruling quotes, or None when it cannot be placed."""
    try:
        broken = _user_citations([raw], turns)
    except _UnusableCheck:
        return None
    return broken[0] if broken else None


def _preemption(
    raw: dict[str, Any], window: str, turns: list[Turn], strategies: list[StrategyVerdict]
) -> dict[str, Any] | None:
    """The preemption fields, when the answer meets the rule; else None (``not_attempted``).

    A preemption needs a multi-turn behavior, a reason, and a verbatim AI turn. A strategy it
    names must be one whose trigger never came.
    """
    if raw.get("miss") != "preempted" or window != "multi_turn":
        return None
    preempted_by = raw.get("preempted_by")
    if preempted_by not in ("resolved", "derailed"):
        return None
    try:
        ai_citation = _citation(raw.get("ai_citation"), turns)
    except _UnusableCheck:
        return None
    if ai_citation is None or ai_citation.speaker != "assistant":
        return None
    index = raw.get("strategy_index")
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if not 1 <= index <= len(strategies):
            return None
        if strategies[index - 1].adherence != "trigger_absent":
            return None
    return {"preempted_by": preempted_by, "ai_citation": ai_citation, "strategy_index": index}


def _unique(citations: Any) -> list[TurnCitation]:
    seen: set[tuple[int, str, str]] = set()
    out: list[TurnCitation] = []
    for c in citations:
        key = (c.turn, c.speaker, c.quote)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return sorted(out, key=lambda c: c.turn)


def _in_window(citations: list[TurnCitation], available_from_turn: int) -> list[TurnCitation]:
    return [
        item for item in citations if item.speaker == "user" and item.turn >= available_from_turn
    ]


def _citations(raw: Any, turns: list[Turn]) -> list[TurnCitation]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _UnusableCheck("citations is not a list")
    found: list[TurnCitation] = []
    for item in raw:
        citation = _citation(item, turns)
        if citation is not None:
            found.append(citation)
    return found


# Models often rewrite apostrophes and dashes. The stored quote is the transcript's
# own characters, so the citation stays verbatim.
_QUOTE_PUNCT = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


def _verbatim_quote(quote: str, content: str) -> str | None:
    if quote in content:
        return quote
    normalized_quote = quote.translate(_QUOTE_PUNCT)
    normalized_content = content.translate(_QUOTE_PUNCT)
    start = normalized_content.find(normalized_quote)
    if start == -1 or not normalized_quote:
        return None
    return content[start : start + len(normalized_quote)]


def _citation(raw: Any, turns: list[Turn]) -> TurnCitation | None:
    """Place one citation on the message it quotes.

    ``speaker`` is required. The user message and the assistant reply share
    ``turn_number``, so a citation without ``"user"`` or ``"assistant"`` cannot
    be placed on one of them. That response is unusable rather than dropped.
    """
    if not isinstance(raw, dict):
        return None
    try:
        turn_number = int(raw["turn"])
    except (KeyError, TypeError, ValueError):
        return None
    speaker = raw.get("speaker")
    if speaker not in ("user", "assistant"):
        raise _UnusableCheck(
            f"citation at turn {turn_number} needs speaker 'user' or 'assistant'; got {speaker!r}"
        )
    match = next(
        (turn for turn in turns if turn.turn_number == turn_number and turn.role == speaker),
        None,
    )
    if match is None:
        return None
    quote = raw.get("quote")
    if not isinstance(quote, str):
        return None
    verbatim = _verbatim_quote(quote, match.content)
    if verbatim is None:
        return None
    try:
        return TurnCitation(turn=turn_number, speaker=speaker, quote=verbatim)
    except ValidationError:
        return None


def _by_index(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for item in items:
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _UnusableCheck("a strategy has no index") from exc
        if index in found:
            raise _UnusableCheck(f"strategy {index} is repeated")
        found[index] = item
    return found


def _by_behavior(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for item in items:
        behavior = item.get("behavior")
        if not isinstance(behavior, str) or not behavior:
            raise _UnusableCheck("a behavior assertion has no behavior id")
        if behavior in found:
            raise _UnusableCheck(f"behavior {behavior} is repeated")
        found[behavior] = item
    return found


def _object_list(parsed: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = parsed.get(key)
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise _UnusableCheck(f"{key} is not a list of objects")
    return raw


def _status(
    behaviors: list[BehaviorVerdict],
) -> tuple[Literal["valid", "invalid", "not_tested"], list[FidelityReason]]:
    reasons = [
        FidelityReason(behavior=item.behavior, miss="not_attempted")
        for item in behaviors
        if item.miss == "not_attempted"
    ]
    if reasons:
        return "invalid", reasons
    if any(item.miss == "preempted" for item in behaviors):
        return "not_tested", []
    return "valid", []


def parse_json_answer(raw: str) -> dict[str, Any] | None:
    """The JSON object a model answered with: the whole reply, its last fenced block, or the
    text from its first ``{`` to its last ``}``."""
    stripped = raw.strip()
    parsed = _load_object(stripped)
    if parsed is not None:
        return parsed
    # A reply that revises its answer holds several fenced blocks; the last one is its answer.
    for block in reversed(re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", stripped)):
        parsed = _load_object(block)
        if parsed is not None:
            return parsed
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return _load_object(stripped[start : end + 1])
    return None


def _load_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
