"""Read the trace files beside results JSONL: the sim driver's prompts, the fidelity check's call
and the judge's calls.

A trace row names its test case but not its transcript, so callers narrow rows by test case,
by the judge run's time window and by the transcript's own text.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sim_trace_path(sim_file: Path) -> Path:
    """The batch trace beside a ``*-sim.jsonl``: ``<stem>-<batch folder>.trace.jsonl``.

    A folder whose only sim trace has another name falls back to that trace. With several, a
    trace not named for this file belongs to another system tested, so none is returned.
    """
    batch = sim_file.parent.name
    stem = sim_file.stem
    for name in (f"{stem}-{batch}", f"{stem.removesuffix('-sim')}-{batch}", stem):
        named = sim_file.parent / f"{name}.trace.jsonl"
        if named.is_file():
            return named
    traces = [
        path
        for path in sim_file.parent.glob("*.trace.jsonl")
        if not path.name.endswith(("-judge.trace.jsonl", "-check.trace.jsonl"))
    ]
    return traces[0] if len(traces) == 1 else sim_file.parent / f"{stem}-{batch}.trace.jsonl"


def check_trace_path(sim_file: Path) -> Path:
    """``<stem>-sim.jsonl`` → ``<stem>-check.trace.jsonl``, written by ``corallab check``."""
    return sim_file.with_name(f"{sim_file.stem.removesuffix('-sim')}-check.trace.jsonl")


def check_call_from_rows(
    rows: list[dict[str, Any]], test_case_id: str, person_text: str
) -> dict[str, Any] | None:
    """The fidelity check's call on one transcript, from its check trace rows.

    Rows name the test case, not the transcript; when several transcripts share the test case,
    the call whose prompt holds ``person_text`` (a person message of this transcript) is it.
    """
    calls = [
        row
        for row in rows
        if row.get("role") == "elicitation_fidelity" and row.get("test_case_id") == test_case_id
    ]
    if len(calls) > 1 and person_text:
        calls = [row for row in calls if person_text in str(row.get("prompt") or "")]
    return calls[-1] if calls else None


def judge_trace_path(judge_file: Path) -> Path:
    """``<stem>-judge.jsonl`` → ``<stem>-judge.trace.jsonl``."""
    return judge_file.with_name(f"{judge_file.stem}.trace.jsonl")


def trace_rows(trace_path: Path, test_case_id: str | None = None) -> list[dict[str, Any]]:
    """Parsed rows of a trace file, only those naming ``test_case_id`` when it is given.

    A row without a ``test_case_id`` field is kept either way. Empty when the file is missing.
    """
    needle = test_case_id.encode("utf-8") if test_case_id else None
    rows: list[dict[str, Any]] = []
    try:
        with trace_path.open("rb") as handle:
            for raw in handle:
                if (
                    needle is not None
                    and needle not in raw
                    and b'"test_case_id"' in raw
                    and b'"test_case_id": null' not in raw
                ):
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _match_row(
    rows: list[dict[str, Any]], role: str, test_case_id: str | None
) -> dict[str, Any] | None:
    """The first row of ``role`` for ``test_case_id``.

    A row without a ``test_case_id`` matches only when no row of that role carries one, so a
    multi-case trace never shows another case's prompt.
    """
    of_role = [row for row in rows if row.get("role") == role]
    if not test_case_id:
        return of_role[0] if of_role else None
    for row in of_role:
        if row.get("test_case_id") == test_case_id:
            return row
    if not any(row.get("test_case_id") for row in of_role):
        return of_role[0] if of_role else None
    return None


def sim_driver_from_rows(
    rows: list[dict[str, Any]], test_case_id: str | None = None
) -> tuple[str | None, str, str]:
    """``(system prompt of the system tested, label, driver prompt)`` for one test case.

    The driver is the persona specification at L3 and the scripted first turn at L1 and L2.
    ``(None, "", "")`` when the trace holds neither.
    """
    sut_row = _match_row(rows, "sut", test_case_id)
    person_row = _match_row(rows, "person_response_generator", test_case_id)
    sut_system = (sut_row.get("system") or None) if sut_row else None
    person_system = (person_row.get("system") or "") if person_row else ""
    if person_system:
        return (sut_system, "Persona Specification", person_system)
    if sut_row is not None:
        return (sut_system, "Scripted Prompt (turn 1)", sut_row.get("prompt") or "")
    return (None, "", "")


@dataclass(frozen=True)
class JudgeCall:
    """One model call of the two-stage judge in a judge trace, located by byte offset."""

    seq: int
    role: str
    stage: int
    label: str
    test_case_id: str
    timestamp: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    offset: int


_JUDGE_ROLE = re.compile(r"^coralmind_s([12])_(.+)$")
_GROUP = re.compile(r"^(ai|user)_behavior-g0*(\d+)$")
_INDEX_LIMIT = 32

_index_lock = threading.Lock()
_indexes: OrderedDict[Path, tuple[tuple[int, int], tuple[JudgeCall, ...]]] = OrderedDict()


def _call_label(stage: int, name: str) -> str:
    group = _GROUP.match(name)
    if stage == 1 and group:
        who = "AI" if group.group(1) == "ai" else "person"
        return f"{who} behaviors, group {group.group(2)}"
    return name


def _number(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _scan(trace_path: Path) -> tuple[JudgeCall, ...]:
    calls: list[JudgeCall] = []
    offset = 0
    with trace_path.open("rb") as handle:
        for raw in handle:
            at, offset = offset, offset + len(raw)
            if b'"llm_call"' not in raw or b"coralmind_s" not in raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            match = _JUDGE_ROLE.match(str(row.get("role") or "")) if isinstance(row, dict) else None
            if match is None:
                continue
            stage = int(match.group(1))
            calls.append(
                JudgeCall(
                    seq=_number(row.get("seq")),
                    role=str(row.get("role")),
                    stage=stage,
                    label=_call_label(stage, match.group(2)),
                    test_case_id=str(row.get("test_case_id") or ""),
                    timestamp=str(row.get("timestamp") or ""),
                    prompt_tokens=_number(row.get("prompt_tokens")),
                    completion_tokens=_number(row.get("completion_tokens")),
                    latency_ms=_number(row.get("latency_ms")),
                    offset=at,
                )
            )
    return tuple(calls)


def judge_trace_index(trace_path: Path) -> tuple[JudgeCall, ...]:
    """Every judge call in a trace file, read once per ``(size, mtime)`` of the file."""
    try:
        stat = trace_path.stat()
    except OSError:
        return ()
    key = (stat.st_size, stat.st_mtime_ns)
    with _index_lock:
        cached = _indexes.get(trace_path)
        if cached is not None and cached[0] == key:
            _indexes.move_to_end(trace_path)
            return cached[1]
    try:
        calls = _scan(trace_path)
    except OSError:
        return ()
    with _index_lock:
        _indexes[trace_path] = (key, calls)
        _indexes.move_to_end(trace_path)
        while len(_indexes) > _INDEX_LIMIT:
            _indexes.popitem(last=False)
    return calls


def judge_calls(
    trace_path: Path,
    test_case_id: str,
    *,
    started: str = "",
    finished: str = "",
    excerpt: str = "",
) -> list[JudgeCall]:
    """The Stage 1 and Stage 2 calls of one judged transcript, in call order.

    Several samples of a case judged in one batch share their rows' test case. Rows are kept
    when their timestamp falls inside the judge run's ``started``–``finished`` window, and a
    role that still repeats is narrowed to the rows whose prompt contains ``excerpt``.
    """
    if not test_case_id:
        return []
    found = [
        call
        for call in judge_trace_index(trace_path)
        if call.test_case_id == test_case_id
        and (not (started and finished) or started <= call.timestamp <= finished)
    ]
    roles = [call.role for call in found]
    if excerpt and len(roles) != len(set(roles)):
        narrowed = [c for c in found if excerpt in trace_text(trace_path, c.offset, "prompt")]
        found = narrowed or found
    return sorted(found, key=lambda c: c.seq)


def trace_row(trace_path: Path, offset: int) -> dict[str, Any]:
    """The trace row at ``offset``; empty when that line is not a JSON object."""
    with trace_path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.readline()
    try:
        row = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return row if isinstance(row, dict) else {}


def trace_text(trace_path: Path, offset: int, part: str) -> str:
    """One field (``prompt``, ``response`` or ``system``) of the trace row at ``offset``."""
    value = trace_row(trace_path, offset).get(part)
    return value if isinstance(value, str) else ""
