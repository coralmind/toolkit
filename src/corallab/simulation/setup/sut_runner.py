"""System-under-test transcript generation for L1/L2 test cases.

L1 and L2 both send every scripted turn in order, with a reply after each.
"""

from __future__ import annotations

import logging

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import (
    DEFAULT_SUT_MAX_TOKENS,
    DEFAULT_SUT_TEMPERATURE,
    RunManifest,
)
from corallab.schemas.eval.transcript import SimFailure

logger = logging.getLogger(__name__)

_MAX_DETAIL_CHARS = 500
_EMPTY_RESPONSE_DETAIL = "SUT returned an empty response (possible content filter or refusal)."


def _classify_exception(exc: Exception) -> str:
    """Map an exception to a SimFailure.reason category.

    Matches on the exception type name so it works whether litellm is imported
    or the error bubbles up as a generic Exception (the names are stable across
    litellm versions: RateLimitError, ContentPolicyViolationError, Timeout, ...).
    """
    name = type(exc).__name__
    text = f"{name} {exc}".lower()
    if "contentpolicyviolation" in text or "content_filter" in text or "content management" in text:
        return "content_filter"
    if "ratelimit" in text or "429" in text:
        return "rate_limit"
    if "timeout" in text:
        return "timeout"
    return "error"


async def call_system_under_test_for_transcript(
    case: RunCase,
    manifest: RunManifest,
    evaluation_id: str,
) -> list[Turn] | SimFailure:
    """Call the system-under-test model to generate a transcript for L1/L2 test cases.

    Sends every scripted turn in order, with a reply after each (L1 and L2 alike).

    Returns list[Turn] on success, or a SimFailure describing why the call
    failed (so the case is preserved in the output rather than dropped).
    """
    from corallab.llm_client import llm_client

    sut = manifest.primary_sut

    record = case.test_case
    scenario_turns = record.turns
    opening = scenario_turns[0]
    conv_turns: list[Turn] = []
    messages: list[dict[str, str]] = []

    conv_turns.append(Turn(turn_number=1, role="user", content=opening))
    messages.append({"role": "user", "content": opening})

    logger.info(
        "Calling system-under-test for L%d transcript: test_case_id=%s model=%s",
        record.evaluation_level,
        case.test_case_id,
        sut.model_id,
    )

    system_under_test_max_tokens = sut.max_tokens_or(DEFAULT_SUT_MAX_TOKENS)
    sut_temperature = sut.temperature_or(DEFAULT_SUT_TEMPERATURE)
    sut_extra_params = sut.completion_params()

    try:
        response = await llm_client.complete(
            model=sut.model_id,
            messages=messages,
            max_tokens=system_under_test_max_tokens,
            temperature=sut_temperature,
            trace_role="sut",
            trace_test_case_id=case.test_case_id,
            extra_params=sut_extra_params,
        )
        if not response.text.strip():
            logger.error(
                "System-under-test returned empty response for test_case_id=%s turn=2 "
                "(finish_reason may be content_filter or length with reasoning tokens). "
                "Aborting — cannot evaluate an empty conversation.",
                case.test_case_id,
            )
            return SimFailure(
                reason="empty_response",
                detail=_EMPTY_RESPONSE_DETAIL,
                failed_at_turn=1,
            )
        conv_turns.append(Turn(turn_number=1, role="assistant", content=response.text))
        messages.append({"role": "assistant", "content": response.text})

        if len(scenario_turns) > 1:
            for exchange, scripted_msg in enumerate(scenario_turns[1:], start=2):
                conv_turns.append(Turn(turn_number=exchange, role="user", content=scripted_msg))
                messages.append({"role": "user", "content": scripted_msg})

                response = await llm_client.complete(
                    model=sut.model_id,
                    messages=messages,
                    max_tokens=system_under_test_max_tokens,
                    temperature=sut_temperature,
                    trace_role="sut",
                    trace_test_case_id=case.test_case_id,
                    extra_params=sut_extra_params,
                )
                if not response.text.strip():
                    logger.error(
                        "System-under-test returned empty response for test_case_id=%s turn=%d "
                        "(finish_reason may be content_filter or length with reasoning tokens). "
                        "Aborting — cannot evaluate an empty conversation.",
                        case.test_case_id,
                        exchange,
                    )
                    return SimFailure(
                        reason="empty_response",
                        detail=_EMPTY_RESPONSE_DETAIL,
                        failed_at_turn=exchange,
                    )
                conv_turns.append(
                    Turn(turn_number=exchange, role="assistant", content=response.text)
                )
                messages.append({"role": "assistant", "content": response.text})

    except Exception as exc:
        reason = _classify_exception(exc)
        logger.error(
            "System-under-test call failed for test_case_id=%s (reason=%s): %s",
            case.test_case_id,
            reason,
            exc,
            exc_info=True,
        )
        user_turns = sum(1 for t in conv_turns if t.role == "user")
        return SimFailure(
            reason=reason,
            detail=str(exc)[:_MAX_DETAIL_CHARS],
            failed_at_turn=user_turns or None,
        )

    logger.info(
        "System-under-test transcript generated: test_case_id=%s turns=%d",
        case.test_case_id,
        len(conv_turns),
    )
    return conv_turns
