"""Tests for PersonaConsistencyChecker quality scoring extension.

Covers:
- _parse_quality_score: valid range, out-of-range, non-numeric, None
- _parse_dimension_scores: valid dict, missing keys, non-numeric values, None/non-dict
- check() returns quality_score and dimension_scores on LLM success
- check() falls back to defaults on LLM failure
- check() falls back to defaults on JSON parse failure
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from corallab.fidelity.consistency_checker import (
    _DEFAULT_DIMENSION_SCORE,
    _DEFAULT_QUALITY_SCORE,
    _QUALITY_DIMENSIONS,
    PersonaConsistencyChecker,
)

# ---------------------------------------------------------------------------
# _parse_quality_score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, 1),
        (3, 3),
        (5, 5),
        ("4", 4),
        (4.9, 4),  # float truncated via int()
        (0, 3),  # out of range → default
        (6, 3),  # out of range → default
        (None, 3),  # None → default
        ("abc", 3),  # non-numeric → default
        ({}, 3),  # wrong type → default
    ],
)
def test_parse_quality_score(raw, expected):
    assert PersonaConsistencyChecker._parse_quality_score(raw) == expected


# ---------------------------------------------------------------------------
# _parse_dimension_scores
# ---------------------------------------------------------------------------


def test_parse_dimension_scores_valid():
    raw = {
        "clinical_accuracy": 4,
        "symptom_consistency": 3,
        "register_appropriateness": 5,
        "behavioral_expression": 4,
    }
    result = PersonaConsistencyChecker._parse_dimension_scores(raw)
    assert result == raw


def test_parse_dimension_scores_all_keys_present():
    raw = dict.fromkeys(_QUALITY_DIMENSIONS, 3)
    result = PersonaConsistencyChecker._parse_dimension_scores(raw)
    assert set(result.keys()) == set(_QUALITY_DIMENSIONS)


def test_parse_dimension_scores_missing_keys_filled_with_default():
    raw = {"clinical_accuracy": 5}  # three keys missing
    result = PersonaConsistencyChecker._parse_dimension_scores(raw)
    assert result["clinical_accuracy"] == 5
    assert result["symptom_consistency"] == _DEFAULT_DIMENSION_SCORE
    assert result["register_appropriateness"] == _DEFAULT_DIMENSION_SCORE
    assert result["behavioral_expression"] == _DEFAULT_DIMENSION_SCORE


def test_parse_dimension_scores_out_of_range_values():
    raw = dict.fromkeys(_QUALITY_DIMENSIONS, 0)  # 0 is out of range
    result = PersonaConsistencyChecker._parse_dimension_scores(raw)
    assert all(v == _DEFAULT_DIMENSION_SCORE for v in result.values())


def test_parse_dimension_scores_non_numeric_values():
    raw = dict.fromkeys(_QUALITY_DIMENSIONS, "great")
    result = PersonaConsistencyChecker._parse_dimension_scores(raw)
    assert all(v == _DEFAULT_DIMENSION_SCORE for v in result.values())


def test_parse_dimension_scores_none_returns_all_defaults():
    result = PersonaConsistencyChecker._parse_dimension_scores(None)
    assert result == dict.fromkeys(_QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE)


def test_parse_dimension_scores_non_dict_returns_all_defaults():
    result = PersonaConsistencyChecker._parse_dimension_scores("oops")
    assert result == dict.fromkeys(_QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE)


# ---------------------------------------------------------------------------
# check() integration: quality fields propagate correctly
# ---------------------------------------------------------------------------


def _make_pcc() -> PersonaConsistencyChecker:
    client = MagicMock()
    return PersonaConsistencyChecker(client=client, model="test-model")


@pytest.mark.asyncio
async def test_check_returns_quality_fields_on_success():
    pcc = _make_pcc()
    response_json = """{
        "consistent": true,
        "violation_type": null,
        "reasoning": "good",
        "quality_score": 4,
        "dimension_scores": {
            "clinical_accuracy": 4,
            "symptom_consistency": 4,
            "register_appropriateness": 5,
            "behavioral_expression": 5
        }
    }"""
    mock_response = MagicMock()
    mock_response.text = response_json
    pcc._client.complete = AsyncMock(return_value=mock_response)

    result = await pcc.check(message="hey i'm feeling anxious", last_3_turns=[])

    assert result["consistent"] is True
    assert result["quality_score"] == 4
    assert result["dimension_scores"]["clinical_accuracy"] == 4
    assert result["dimension_scores"]["behavioral_expression"] == 5


@pytest.mark.asyncio
async def test_check_quality_defaults_on_llm_failure():
    pcc = _make_pcc()
    pcc._client.complete = AsyncMock(side_effect=Exception("LLM down"))

    result = await pcc.check(message="hey", last_3_turns=[])

    assert result["quality_score"] == _DEFAULT_QUALITY_SCORE
    assert result["dimension_scores"] == dict.fromkeys(
        _QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE
    )
    assert result["consistent"] is True  # fail-safe


@pytest.mark.asyncio
async def test_check_quality_defaults_on_json_parse_failure():
    pcc = _make_pcc()
    mock_response = MagicMock()
    mock_response.text = "this is not json"
    pcc._client.complete = AsyncMock(return_value=mock_response)

    result = await pcc.check(message="hey", last_3_turns=[])

    assert result["quality_score"] == _DEFAULT_QUALITY_SCORE
    assert result["dimension_scores"] == dict.fromkeys(
        _QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE
    )
    assert result["consistent"] is True  # fail-safe


@pytest.mark.asyncio
async def test_check_quality_defaults_when_fields_missing_from_response():
    """LLM returns valid JSON but omits quality fields — should fall back to defaults."""
    pcc = _make_pcc()
    mock_response = MagicMock()
    mock_response.text = '{"consistent": true, "violation_type": null, "reasoning": "ok"}'
    pcc._client.complete = AsyncMock(return_value=mock_response)

    result = await pcc.check(message="hey", last_3_turns=[])

    assert result["quality_score"] == _DEFAULT_QUALITY_SCORE
    assert result["dimension_scores"] == dict.fromkeys(
        _QUALITY_DIMENSIONS, _DEFAULT_DIMENSION_SCORE
    )
