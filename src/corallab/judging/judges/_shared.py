"""Helpers the two-stage judge shares: refusal detection, JSON parsing, transcript
formatting, the person-context block, and the process-wide LLM semaphore."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.persona import Persona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = [
    "i cannot evaluate",
    "i can't evaluate",
    "i am unable to evaluate",
    "i'm unable to evaluate",
    "i cannot assist",
    "i can't assist",
    "i will not evaluate",
    "i won't evaluate",
]


def is_refusal(text: str) -> bool:
    """Check whether a response text looks like a refusal."""
    lower = text.lower().strip()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _fix_escaped_delimiters(text: str) -> str:
    """Fix LLM output where \\" is used as a string delimiter instead of ".

    Some models emit `"key": \\"value\\"` instead of `"key": "value"`.
    This replaces `\\"` only at value-boundary positions (after `: ` or `[`)
    and before `,` `}` `]` or end-of-line, leaving mid-string `\\"` untouched.
    """
    text = re.sub(r"(:\s*)\\\"", r'\1"', text)
    text = re.sub(r"\\\"(?=\s*[,\}\]\n])", '"', text)
    return text


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    """Attempt to parse JSON from a judge response.

    Tries direct parse, then falls back to extracting JSON from markdown fences,
    then brace-scan, then a cleanup pass for escaped-delimiter artifacts.
    Returns None on failure.
    """
    stripped = raw.strip()

    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]+\})\s*```", stripped)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        fixed = _fix_escaped_delimiters(candidate)
        if fixed != candidate:
            try:
                parsed = json.loads(fixed)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass

    return None


# ---------------------------------------------------------------------------
# Clinical context
# ---------------------------------------------------------------------------


def _append_person_context(
    parts: list[str], persona: Persona | None, heading: str = "## Clinical Context\n"
) -> None:
    """Append the person context block (description and clinical portrait) to ``parts``.

    Only ``persona_description`` and ``clinical_context`` reach context-informed
    judges: no test case metadata, no traits or states.
    """
    if persona is None:
        return
    if not persona.persona_description and not persona.clinical_context:
        return
    parts.append(heading)
    if persona.persona_description:
        parts.append(f"**About This Person:**\n{persona.persona_description}")
    if persona.clinical_context:
        parts.append(f"**Clinical Background:**\n{persona.clinical_context}")


# ---------------------------------------------------------------------------
# Transcript formatter
# ---------------------------------------------------------------------------


def _format_transcript(turns: list[Turn] | None) -> str:
    if turns is None:
        msg = (
            "conversation_transcript is None — cannot format transcript. "
            "Ensure the test case has been through SUT or simulation first."
        )
        raise ValueError(msg)
    lines = ["## Conversation Transcript", ""]
    for turn in turns:
        role_label = "USER" if turn.role == "user" else "AI ASSISTANT"
        lines.append(f"**Turn {turn.turn_number} — {role_label}:**")
        lines.append(turn.content)
        lines.append("")
    return "\n".join(lines)


# One semaphore for the process. MapStates fans out test cases, and each case
# fans out its own calls; a per-case cap would not bound the batch.
_DEFAULT_MAX_CONCURRENT_LLM_CALLS = 8
_MAX_CONCURRENCY_ENV_VAR = "CORALMIND_JUDGE_MAX_CONCURRENCY"


def get_max_concurrent_llm_calls() -> int:
    """Read ``CORALMIND_JUDGE_MAX_CONCURRENCY``, or the default.

    Missing, empty, non-integer, or values below 1 use the default.
    ``Semaphore(0)`` would deadlock every judge call.
    """
    raw = os.environ.get(_MAX_CONCURRENCY_ENV_VAR)
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_CONCURRENT_LLM_CALLS
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d",
            _MAX_CONCURRENCY_ENV_VAR,
            raw,
            _DEFAULT_MAX_CONCURRENT_LLM_CALLS,
        )
        return _DEFAULT_MAX_CONCURRENT_LLM_CALLS
    if value < 1:
        logger.warning(
            "%s=%d is below the minimum of 1; using default %d",
            _MAX_CONCURRENCY_ENV_VAR,
            value,
            _DEFAULT_MAX_CONCURRENT_LLM_CALLS,
        )
        return _DEFAULT_MAX_CONCURRENT_LLM_CALLS
    return value


_llm_semaphore: asyncio.Semaphore | None = None
_llm_semaphore_loop: asyncio.AbstractEventLoop | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the shared judge LLM semaphore, bound to the running event loop.

    An ``asyncio.Semaphore`` attaches its internal waiter futures to whichever
    loop first blocks on it, so a semaphore cached from a prior loop raises
    ``RuntimeError: got Future attached to a different loop`` when reused under a
    new loop (e.g. ``corallab run`` drives sim then judge as two sequential
    ``asyncio.run`` calls; pytest runs each async test on a fresh loop). We track
    the loop the semaphore was created on and re-create it when the running loop
    changes, so the cap stays a single process-global without leaking across loops.
    """
    global _llm_semaphore, _llm_semaphore_loop
    running_loop = asyncio.get_running_loop()
    if _llm_semaphore is None or _llm_semaphore_loop is not running_loop:
        _llm_semaphore = asyncio.Semaphore(get_max_concurrent_llm_calls())
        _llm_semaphore_loop = running_loop
    return _llm_semaphore
