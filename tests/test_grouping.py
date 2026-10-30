"""Groups of test cases, and the directories a sim run writes them into."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _records import make_persona, make_test_case, write_test_case_records

from corallab.cli._runners import run_batch_sim
from corallab.grouping import iter_groups
from corallab.llm_client import LLMResponse
from corallab.schemas.eval.run_manifest import RunManifest

_JUDGE = {
    "judge_type": "kl_two_stage",
    "model": {"model_id": "test-judge"},
}


def test_iter_groups_splits_the_tail() -> None:
    groups = list(iter_groups(["a", "b", "c", "d", "e"], 2))
    assert groups == [(1, ["a", "b"]), (2, ["c", "d"]), (3, ["e"])]


def test_iter_groups_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_groups([1], 0))


async def _ok(**kwargs: Any) -> LLMResponse:
    return LLMResponse(
        text="ok",
        model="mock",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0},
    )


async def test_sim_writes_each_group_to_its_own_directory(tmp_path: Path) -> None:
    cases = [make_test_case(f"aimhef-0000000{i}", evaluation_level=1) for i in range(1, 4)]
    records_path, personas = write_test_case_records(tmp_path, cases, [make_persona()])
    requested = tmp_path / "out" / "run-sim.jsonl"
    manifest = RunManifest(
        run_id="groups",
        evaluation_level=1,
        system_under_test=[{"model_id": "mock/sut"}],
        test_case_records_path=str(records_path),
        persona_records_path=str(personas),
        run_dir=str(requested.parent),
        simulation={"batch_size": 2},
        retry_max=0,
        retry_backoff_ms=0,
        judges=[_JUDGE],
    )

    with patch("corallab.llm_client.llm_client.complete", new=_ok):
        outputs, loaded, _, written = await run_batch_sim(
            manifest, str(records_path), requested, None, "run", dry_run=False
        )

    assert len(loaded) == 3
    assert len(outputs) == 3
    assert written == [
        requested.parent / "batch-001" / "run-sim.jsonl",
        requested.parent / "batch-002" / "run-sim.jsonl",
    ]
    assert not requested.exists()
    first = (requested.parent / "batch-001" / "run-sim.jsonl").read_text().strip().splitlines()
    second = (requested.parent / "batch-002" / "run-sim.jsonl").read_text().strip().splitlines()
    assert len(first) == 2
    assert len(second) == 1
    assert "aimhef-00000001" in first[0]
    assert "aimhef-00000003" in second[0]
