"""Trace readers: the sim driver's prompts and the judge's calls."""

from __future__ import annotations

import json
from pathlib import Path

from coralmind_viewer.trace_loaders import (
    judge_calls,
    judge_trace_index,
    sim_driver_from_rows,
    sim_trace_path,
    trace_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_the_scripted_driver_is_the_cases_own_turn(tmp_path: Path) -> None:
    trace = tmp_path / "run-sim.trace.jsonl"
    _write_jsonl(
        trace,
        [
            {"role": "sut", "test_case_id": "aimhef-ocd00001", "system": "A", "prompt": "turn A"},
            {"role": "sut", "test_case_id": "aimhef-ssh00001", "system": "B", "prompt": "turn B"},
        ],
    )
    rows = trace_rows(trace, "aimhef-ssh00001")
    assert sim_driver_from_rows(rows, "aimhef-ssh00001") == (
        "B",
        "Scripted Prompt (turn 1)",
        "turn B",
    )


def test_the_persona_specification_wins_over_the_scripted_turn(tmp_path: Path) -> None:
    trace = tmp_path / "run-sim.trace.jsonl"
    _write_jsonl(
        trace,
        [
            {"role": "sut", "test_case_id": "aimhef-a0000001", "system": "SUT", "prompt": "hi"},
            {
                "role": "person_response_generator",
                "test_case_id": "aimhef-a0000001",
                "system": "You are Dana.",
            },
        ],
    )
    rows = trace_rows(trace, "aimhef-a0000001")
    assert sim_driver_from_rows(rows, "aimhef-a0000001") == (
        "SUT",
        "Persona Specification",
        "You are Dana.",
    )


def test_a_row_without_a_case_id_is_used_only_when_no_row_names_one(tmp_path: Path) -> None:
    old = [{"role": "sut", "system": "S", "prompt": "only"}]
    assert sim_driver_from_rows(old, "aimhef-a0000001")[2] == "only"
    mixed = [*old, {"role": "sut", "test_case_id": "aimhef-b0000001", "prompt": "b"}]
    assert sim_driver_from_rows(mixed, "aimhef-a0000001") == (None, "", "")


def test_a_missing_trace_reads_as_empty(tmp_path: Path) -> None:
    assert trace_rows(tmp_path / "missing.trace.jsonl") == []
    assert sim_driver_from_rows([], "aimhef-a0000001") == (None, "", "")


def _call(
    seq: int, case: str, stamp: str, prompt: str, role: str = "coralmind_s1_ai_behavior-g01"
) -> dict:
    return {
        "type": "llm_call",
        "seq": str(seq),
        "role": role,
        "test_case_id": case,
        "timestamp": stamp,
        "prompt": prompt,
        "response": "r",
        "prompt_tokens": "9",
    }


def test_judge_calls_are_indexed_once_per_file_version(tmp_path: Path) -> None:
    trace = tmp_path / "run-judge.trace.jsonl"
    _write_jsonl(trace, [_call(1, "aimhef-a0000001", "2026-10-20T20:00:01", "p")])
    first = judge_trace_index(trace)
    assert judge_trace_index(trace) is first
    _write_jsonl(
        trace,
        [
            _call(1, "aimhef-a0000001", "2026-10-20T20:00:01", "p"),
            _call(2, "aimhef-a0000001", "2026-10-20T20:00:02", "q"),
        ],
    )
    assert len(judge_trace_index(trace)) == 2


def test_judge_calls_narrow_by_window_then_by_text(tmp_path: Path) -> None:
    trace = tmp_path / "run-judge.trace.jsonl"
    _write_jsonl(
        trace,
        [
            {"type": "event", "seq": "0", "role": "action_load"},
            _call(1, "aimhef-a0000001", "2026-10-20T20:00:01", "sample one text"),
            _call(2, "aimhef-a0000001", "2026-10-20T20:00:02", "sample two text"),
            _call(3, "aimhef-a0000001", "2026-10-20T20:05:00", "later run"),
        ],
    )
    window = {"started": "2026-10-20T20:00:00", "finished": "2026-10-20T20:01:00"}
    both = judge_calls(trace, "aimhef-a0000001", **window)
    assert [c.seq for c in both] == [1, 2]
    one = judge_calls(trace, "aimhef-a0000001", excerpt="sample two", **window)
    assert [(c.seq, c.label) for c in one] == [(2, "AI behaviors, group 1")]


def test_each_system_reads_its_own_batch_trace(tmp_path: Path) -> None:
    batch = tmp_path / "batch-001"
    batch.mkdir()
    for sut in ("luna", "sonnet"):
        (batch / f"run__{sut}-sim-batch-001.trace.jsonl").write_text("", encoding="utf-8")
    (batch / "run__luna-judge.trace.jsonl").write_text("", encoding="utf-8")
    assert sim_trace_path(batch / "run__sonnet-sim.jsonl").name == (
        "run__sonnet-sim-batch-001.trace.jsonl"
    )


def test_another_systems_trace_is_never_read(tmp_path: Path) -> None:
    batch = tmp_path / "batch-001"
    batch.mkdir()
    for name in ("a", "b"):
        (batch / f"{name}.trace.jsonl").write_text("", encoding="utf-8")
    assert not sim_trace_path(batch / "run__sonnet-sim.jsonl").is_file()
