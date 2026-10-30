"""Persona Consistency Checker — validates generated message against persona.

Mode: strict (re-sample up to 3 times on violation).
Returns both binary consistency gate AND per-turn quality score (1-5) with
four clinical dimensions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import litellm

from corallab.llm_client import LLMClient
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.persona import Persona

logger = logging.getLogger(__name__)

_PCC_MAX_TOKENS = 512
_DEFAULT_PCC_TEMPERATURE = 0.0

_VIOLATION_TYPES = [
    "out_of_character",
    "clinically_implausible",
    "breaks_severity_level",
    "anachronistic_knowledge",
]

_QUALITY_DIMENSIONS = [
    "clinical_accuracy",
    "symptom_consistency",
    "register_appropriateness",
    "behavioral_expression",
]

_DEFAULT_QUALITY_SCORE = 3
_DEFAULT_DIMENSION_SCORE = 3


class PersonaConsistencyChecker:
    """Validates generated persona messages against the persona specification.

    Persona traits are read from the supplied ``persona`` (a :class:`Persona`).
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = "claude-haiku-4-5-20251001",
        retry_max: int = 2,
        retry_backoff_ms: int = 1000,
        persona: Persona | None = None,
        temperature: float = _DEFAULT_PCC_TEMPERATURE,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._persona = persona
        self._model = model
        self._retry_max = retry_max
        self._retry_backoff_ms = retry_backoff_ms
        self._temperature = temperature
        self._extra_params = extra_params or {}

    async def check(
        self,
        message: str,
        last_3_turns: list[Turn],
    ) -> dict[str, Any]:
        """Check if a generated message is consistent with the persona.

        Args:
            message: The generated persona message to check.
            last_3_turns: Last 3 turns of conversation for context.

        Returns:
            dict with keys:
              - consistent (bool)
              - violation_type (str | None)
              - reasoning (str | None)
              - quality_score (int 1-5): composite quality score; defaults to 3 on parse failure
              - dimension_scores (dict[str, int]): scores for each of the 4 quality dimensions
        """
        prompt = self._build_check_prompt(message, last_3_turns)
        raw = await self._call_with_retry(prompt)

        if raw is None:
            logger.warning("PCC call failed — accepting message as best attempt")
            return {
                "consistent": True,
                "violation_type": None,
                "reasoning": "PCC unavailable",
                "quality_score": _DEFAULT_QUALITY_SCORE,
                "dimension_scores": dict.fromkeys(_QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE),
            }

        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning("PCC JSON parse failure — accepting message as best attempt")
            return {
                "consistent": True,
                "violation_type": None,
                "reasoning": "PCC parse failure",
                "quality_score": _DEFAULT_QUALITY_SCORE,
                "dimension_scores": dict.fromkeys(_QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE),
            }

        consistent = bool(parsed.get("consistent", True))
        violation_type = parsed.get("violation_type") or None
        reasoning = parsed.get("reasoning") or None

        if violation_type and violation_type not in _VIOLATION_TYPES:
            logger.warning("PCC returned unknown violation_type: %s", violation_type)
            violation_type = "out_of_character"

        if not consistent:
            logger.debug("PCC violation: %s — %s", violation_type, reasoning)

        quality_score = self._parse_quality_score(parsed.get("quality_score"))
        dimension_scores = self._parse_dimension_scores(parsed.get("dimension_scores"))

        return {
            "consistent": consistent,
            "violation_type": violation_type,
            "reasoning": reasoning,
            "quality_score": quality_score,
            "dimension_scores": dimension_scores,
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_check_prompt(
        self,
        message: str,
        last_3_turns: list[Turn],
    ) -> str:
        # Validate against the SAME persona information the Presentation Engine
        # conditions on: the three prose fields (persona_description,
        # clinical_context, behavioral_description). See
        # ``PersonResponseGenerator._part1_persona_identity``.
        p = self._persona
        spec_lines: list[str] = []
        if p and p.persona_description:
            spec_lines.append(f"PERSONA DESCRIPTION:\n{p.persona_description}")
        if p and p.clinical_context:
            spec_lines.append(f"CLINICAL CONTEXT:\n{p.clinical_context}")
        if p and p.behavioral_description:
            spec_lines.append(f"BEHAVIORAL STYLE:\n{p.behavioral_description}")
        persona_spec = "\n\n".join(spec_lines) if spec_lines else "(no persona specification)"

        recent_ctx = "\n".join(f"{t.role.upper()}: {t.content[:200]}" for t in last_3_turns)

        return f"""You are a clinical consistency checker for an AI persona simulation.

TASK: Determine if the generated message is consistent with the persona specification,
and score its clinical quality on four dimensions (1-5 each).

PERSONA SPECIFICATION:
{persona_spec}

RECENT CONVERSATION CONTEXT:
{recent_ctx if recent_ctx else "(no prior turns)"}

GENERATED MESSAGE TO CHECK:
{message}

VIOLATION TYPES (use exactly these strings if violated):
- out_of_character: Message doesn't match the persona's established personality/behavior
- clinically_implausible: Clinical presentation is inconsistent with stated condition/severity
- breaks_severity_level: Distress level is inconsistent with the stated severity band
- anachronistic_knowledge: Persona uses clinical terminology they shouldn't know (e.g., "my OCD is acting up")

QUALITY DIMENSIONS — score each 1-5:

clinical_accuracy (does the response reflect clinically plausible behavior for this condition/severity?):
  5=fully accurate, 4=mostly accurate minor deviation, 3=recognizable but some implausibility,
  2=notable clinical drift, 1=clinically implausible or wrong condition presentation

symptom_consistency (are symptoms/distress consistent with persona spec AND prior 3 turns?):
  5=fully consistent with spec and prior turns, 4=mostly consistent minor drift,
  3=some inconsistency not a clear contradiction, 2=contradicts prior turns,
  1=direct contradiction with persona spec

register_appropriateness (does communication style match stated register?):
  5=exact match (colloquial/hedging/natural disfluencies), 4=mostly matches minor formality drift,
  3=some mismatch slightly too formal, 2=clear violation academic vocabulary,
  1=completely wrong register sounds like different person

behavioral_expression (does the persona express state through natural behavior rather than naming it clinically? "show don't tell"):
  Platform instruction: "Express your STATE through NATURAL BEHAVIOR. Do NOT name your psychological state or use clinical language. Do NOT say 'I feel anxious' — instead, show anxiety through how you write. Do NOT break character. Do NOT narrate what you are doing."
  5=state fully embodied through behavior, no clinical naming, no meta-narration, character never broken,
  4=mostly behavioral, one minor slip (hedged feeling label) but in character,
  3=mixed, partially behavioral but some direct labeling or narration,
  2=frequent clinical naming or out-of-character narration (e.g. "I feel anxious about my OCD"),
  1=complete violation: names diagnosis, uses clinical vocabulary, or breaks character entirely

INSTRUCTIONS:
- Be lenient for minor stylistic deviations; only flag clear violations.
- The persona uses lowercase, hedging language, and colloquial style.
- The persona should NOT name their own psychological state clinically.

Respond ONLY with valid JSON:
{{
  "consistent": true/false,
  "violation_type": null or one of the violation type strings above,
  "reasoning": "brief explanation (1-2 sentences)",
  "quality_score": <integer 1-5, overall composite quality>,
  "dimension_scores": {{
    "clinical_accuracy": <1-5>,
    "symptom_consistency": <1-5>,
    "register_appropriateness": <1-5>,
    "behavioral_expression": <1-5>
  }}
}}"""

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    async def _call_with_retry(self, prompt: str) -> str | None:
        max_attempts = self._retry_max + 1
        backoff_ms = self._retry_backoff_ms
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.complete(
                    model=self._model,
                    max_tokens=_PCC_MAX_TOKENS,
                    temperature=self._temperature,
                    messages=[{"role": "user", "content": prompt}],
                    trace_role="persona_consistency_checker",
                    extra_params=self._extra_params,
                )
                raw_text = response.text
                return raw_text

            except litellm.RateLimitError as exc:
                last_error = exc
                logger.warning("PCC rate limit (attempt %d/%d): %s", attempt, max_attempts, exc)
            except litellm.APIError as exc:
                last_error = exc
                logger.warning("PCC API error (attempt %d/%d): %s", attempt, max_attempts, exc)
            except litellm.APIConnectionError as exc:
                last_error = exc
                logger.warning(
                    "PCC connection error (attempt %d/%d): %s", attempt, max_attempts, exc
                )
            except Exception as exc:
                last_error = exc
                logger.error("PCC unexpected error: %s", exc, exc_info=True)

            if attempt < max_attempts:
                wait_s = (backoff_ms * (2 ** (attempt - 1))) / 1000.0
                await asyncio.sleep(wait_s)

        logger.error("PCC all attempts exhausted. Last error: %s", last_error)
        return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_quality_score(raw_score: Any) -> int:
        """Parse quality_score from PCC response. Defaults to 3 on any failure."""
        try:
            score = int(raw_score)
            if 1 <= score <= 5:
                return score
            logger.warning(
                "PCC quality_score out of range: %s — defaulting to %d",
                raw_score,
                _DEFAULT_QUALITY_SCORE,
            )
        except (TypeError, ValueError):
            if raw_score is not None:
                logger.warning(
                    "PCC quality_score unparseable: %r — defaulting to %d",
                    raw_score,
                    _DEFAULT_QUALITY_SCORE,
                )
        return _DEFAULT_QUALITY_SCORE

    @staticmethod
    def _parse_dimension_scores(raw_dims: Any) -> dict[str, int]:
        """Parse dimension_scores from PCC response. Fills missing/invalid keys with 3."""
        result: dict[str, int] = {}
        if isinstance(raw_dims, dict):
            for dim in _QUALITY_DIMENSIONS:
                raw = raw_dims.get(dim)
                if raw is None:
                    result[dim] = _DEFAULT_DIMENSION_SCORE
                    continue
                try:
                    score = int(raw)
                    result[dim] = score if 1 <= score <= 5 else _DEFAULT_DIMENSION_SCORE
                except (TypeError, ValueError):
                    logger.warning(
                        "PCC dimension_score[%s] unparseable: %r — defaulting to %d",
                        dim,
                        raw,
                        _DEFAULT_DIMENSION_SCORE,
                    )
                    result[dim] = _DEFAULT_DIMENSION_SCORE
        else:
            if raw_dims is not None:
                logger.warning(
                    "PCC dimension_scores missing or invalid — defaulting all to %d",
                    _DEFAULT_DIMENSION_SCORE,
                )
            result = dict.fromkeys(_QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE)
        return result

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        stripped = raw.strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        fence = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", stripped)
        if fence:
            try:
                parsed = json.loads(fence.group(1))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        return None
