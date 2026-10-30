"""Read a results directory into the two tables every viewer number is counted over.

``transcripts`` has one row per transcript record, whatever happened to it; ``routes`` has
one row per judged transcript and route of its suite. Records are read as named fields from
JSON, so runs written by older toolkit versions stay readable, and a line missing a field the
viewer needs is counted as unreadable, never dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from corallab.paths import resolve_data_path
from corallab.simulation.common.copied_messages import copied_ai_message
from coralmind_viewer.catalog import Batch, scan

logger = logging.getLogger(__name__)

BUCKETS = (
    "judged",
    "not_tested",
    "invalid",
    "unchecked",
    "not_judged",
    "sim_failure",
    "unreadable",
)
"""Outcome buckets in ledger order. Every transcript lands in exactly one."""

FACT_MODE_UNRECORDED = "not recorded"

UNKNOWN_SUT = "unknown"
"""The system of a line that names none and comes from a run with several."""

_SUPPORT_RANK = {
    "supported": 3,
    "partially_supported": 2,
    "insufficient_evidence": 1,
    "not_supported": 0,
}
_GATE_APPLIES = ("met", "no_gates")


class Unreadable(ValueError):
    """A record line that lacks a field the viewer needs."""


class StaleRecord(Exception):
    """A file changed since the snapshot was read: its stored offset holds another record."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimRow:
    """One line of a ``*-sim.jsonl`` file: a transcript record, or a line that could not be read.

    An unreadable line has ``error`` set and a ``transcript_id`` naming its file and line.
    """

    run: str
    batch: str
    file: Path
    line: int
    offset: int
    transcript_id: str
    test_case_id: str = ""
    sut_key: str = ""
    exchanges: int = 0
    exit_reason: str | None = None
    sim_failure: str | None = None
    fidelity: str | None = None
    fidelity_reasons: tuple[tuple[str, str], ...] = ()
    preempted_by: str | None = None
    adherence: tuple[str, ...] = ()
    fidelity_turns: tuple[int, ...] = ()
    copied_turn: int | None = None
    error: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        """Transcript ids repeat across run directories made from one manifest."""
        return (self.run, self.transcript_id)


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate of a route, as the judge read it."""

    part_id: str
    label: str
    verdict: str
    conflict: bool


@dataclass(frozen=True, slots=True)
class RouteResult:
    """The judge's Stage 2 verdict on one route of one transcript."""

    pathway_id: str
    did: str
    label: str
    support: str
    gate_status: str
    fires: bool
    partial: bool
    connected: bool
    onset: int | None
    step_turns: tuple[int, ...]
    gates: tuple[GateResult, ...]

    @property
    def outcome(self) -> str:
        """``fires``, ``partial``, ``gate_unknown`` or ``not``."""
        if self.gate_status == "unknown":
            return "gate_unknown"
        if self.fires:
            return "fires"
        return "partial" if self.partial else "not"


@dataclass(frozen=True, slots=True)
class JudgeRow:
    """One line of a ``*-judge.jsonl`` file. ``ok`` is a successful judge run."""

    run: str
    file: Path
    line: int
    offset: int
    transcript_id: str
    evaluated_at: str
    sut_model: str
    judge_model: str
    prompt_version: str
    prompt_hash: str
    kl_digest: str
    kl_version: str
    ok: bool
    partial: bool
    routes: Mapping[str, RouteResult]
    skipped: frozenset[str]
    failed: frozenset[str]
    failed_behaviors: frozenset[str]
    observed: frozenset[str]
    stage1_turns: tuple[int, ...]
    tokens: tuple[int, int, int]
    elapsed_ms: int
    call_failures: int
    retries: int
    mode: str = "live"

    @property
    def key(self) -> tuple[str, str]:
        return (self.run, self.transcript_id)

    @property
    def has_meta(self) -> bool:
        """False for a record whose judge never ran: it belongs to no judge setup."""
        return bool(self.prompt_hash or self.kl_digest)

    @property
    def pass_key(self) -> str:
        parts = [self.judge_model, self.prompt_hash, self.kl_digest]
        if self.mode != "live":
            parts.append(self.mode)
        return "|".join(parts)

    @property
    def firing(self) -> frozenset[str]:
        return frozenset(pid for pid, result in self.routes.items() if result.outcome == "fires")


@dataclass(frozen=True, slots=True)
class ReadProblem:
    """A judge line that could not be read, or a file that could not be opened."""

    file: Path
    line: int
    error: str


def _lines(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    """``(line number, byte offset, raw line)`` for each non-blank line."""
    offset = 0
    for number, raw in enumerate(data.split(b"\n"), start=1):
        if raw.strip():
            yield number, offset, raw
        offset += len(raw) + 1


def _record(raw: bytes) -> dict[str, Any]:
    try:
        rec = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Unreadable(f"not JSON: {exc.msg if hasattr(exc, 'msg') else exc}") from exc
    if not isinstance(rec, dict):
        raise Unreadable("not a JSON object")
    return rec


def _required_str(rec: Mapping[str, Any], key: str) -> str:
    value = rec.get(key)
    if not isinstance(value, str) or not value:
        raise Unreadable(f"missing {key}")
    return value


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def dict_of(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts_of(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _citation_turns(items: Iterable[dict[str, Any]]) -> set[int]:
    turns: set[int] = set()
    for item in items:
        for cite in dicts_of(item.get("citations")):
            turn = cite.get("turn")
            if isinstance(turn, int) and turn >= 1:
                turns.add(turn)
    return turns


def _reason(item: Mapping[str, Any]) -> tuple[str, str]:
    """``(category, detail)``: a Step 6 reason, or a behavior the simulator did not attempt."""
    behavior = _str(item.get("behavior"))
    category = _str(item.get("reason")) or _str(item.get("kind"))
    if not category:
        category = "simulator" if item.get("miss") == "not_attempted" else "other"
    detail = behavior or _str(item.get("detail"))
    return category, detail


def _verdict_fields(
    fidelity: Any,
) -> tuple[str | None, tuple[tuple[str, str], ...], str | None, tuple[str, ...], tuple[int, ...]]:
    """A fidelity verdict's status, reasons, preemption, strategy adherence and cited turns."""
    if not isinstance(fidelity, dict):
        return None, (), None, (), ()
    behaviors = dicts_of(fidelity.get("behaviors"))
    strategies = dicts_of(fidelity.get("strategies"))
    preempted = next(
        (_str(b.get("preempted_by")) for b in behaviors if b.get("miss") == "preempted"), None
    )
    cited = _citation_turns(behaviors)
    cited |= _citation_turns(p for b in behaviors for p in dicts_of(b.get("parts")))
    cited |= _citation_turns(strategies)
    cited |= _citation_turns(dicts_of(fidelity.get("facts")))
    return (
        _str(fidelity.get("status")) or None,
        tuple(_reason(item) for item in dicts_of(fidelity.get("reasons"))),
        preempted,
        tuple(_str(item.get("adherence")) for item in strategies),
        tuple(sorted(cited)),
    )


def with_check(sim: SimRow, verdict: Mapping[str, Any] | None) -> SimRow:
    """``sim`` with the verdict from the fidelity check file beside it; unchanged without one."""
    if verdict is None or sim.error is not None:
        return sim
    status, reasons, preempted, adherence, fidelity_turns = _verdict_fields(verdict)
    return replace(
        sim,
        fidelity=status,
        fidelity_reasons=sim.fidelity_reasons + reasons,
        preempted_by=preempted,
        adherence=adherence,
        fidelity_turns=fidelity_turns,
    )


def parse_sim(run: str, batch: str, file: Path, line: int, offset: int, raw: bytes) -> SimRow:
    """A transcript row. Raises :class:`Unreadable` when a needed field is missing."""
    rec = _record(raw)
    transcript_id = _required_str(rec, "transcript_id")
    test_case_id = _required_str(rec, "test_case_id")
    turns = dicts_of(rec.get("turns"))
    failure = rec.get("sim_failure")
    if isinstance(failure, dict):
        sim_failure: str | None = _str(failure.get("reason")) or "error"
    elif failure is not None:
        sim_failure = "error"
    else:
        sim_failure = None if rec.get("turns") is not None else "error"
    status, reasons, preempted, adherence, fidelity_turns = _verdict_fields(rec.get("fidelity"))
    exchanges = max((_int(t.get("turn_number")) for t in turns), default=0)
    copied = _copied_turn(turns)
    if copied is not None:
        copy = ("simulator", f"the simulated person copied the AI's message at turn {copied}")
        reasons = (copy, *reasons)
    return SimRow(
        run=run,
        batch=batch,
        file=file,
        line=line,
        offset=offset,
        transcript_id=transcript_id,
        test_case_id=test_case_id,
        sut_key=_str(rec.get("system_under_test")),
        exchanges=exchanges,
        exit_reason=_str(rec.get("exit_reason")) or None,
        sim_failure=sim_failure,
        fidelity=status,
        fidelity_reasons=reasons,
        preempted_by=preempted,
        adherence=adherence,
        fidelity_turns=fidelity_turns,
        copied_turn=copied,
    )


def _copied_turn(turns: list[dict[str, Any]]) -> int | None:
    """The first person turn that copies an AI turn before it, or None."""
    history: list[tuple[str, str]] = []
    for item in turns:
        role = _str(item.get("role"))
        text = _str(item.get("content"))
        if role == "user" and copied_ai_message(text, history) is not None:
            return _int(item.get("turn_number"))
        history.append((role, text))
    return None


def route_result(assessment: Mapping[str, Any]) -> RouteResult:
    """A Stage 2 assessment read into a :class:`RouteResult`."""
    instances = dicts_of(assessment.get("instances"))
    supports = [_str(i.get("support")) for i in instances]
    overall = _str(assessment.get("support"), "not_supported")
    best = max(supports, key=lambda s: _SUPPORT_RANK.get(s, -1)) if supports else overall
    gate_status = _str(assessment.get("gate_status"), "no_gates")
    gate_applies = gate_status in _GATE_APPLIES
    supported = [i for i in instances if i.get("support") == "supported"]
    has_supported = bool(supported) or (not instances and overall == "supported")
    fires = has_supported and gate_applies
    partial = not has_supported and gate_applies and best == "partially_supported"
    onset_turns = [
        _int(step.get("turn"))
        for inst in supported
        for step in dicts_of(inst.get("steps"))
        if _int(step.get("turn")) >= 1
    ]
    if not onset_turns:
        onset_turns = [
            _int(inst["turn_window"][0])
            for inst in supported
            if isinstance(inst.get("turn_window"), list) and inst["turn_window"]
        ]
    shown = [i for i in instances if i.get("support") in ("supported", "partially_supported")]
    step_turns = {
        _int(step.get("turn"))
        for inst in shown
        for step in dicts_of(inst.get("steps"))
        if _int(step.get("turn")) >= 1
    }
    gates = tuple(
        GateResult(
            part_id=_str(g.get("part_id")),
            label=_str(g.get("label")),
            verdict=_str(g.get("verdict"), "unknown"),
            conflict=bool(g.get("conflict")),
        )
        for g in dicts_of(assessment.get("gates"))
    )
    return RouteResult(
        pathway_id=_str(assessment.get("pathway_id")),
        did=_str(assessment.get("pathway_did")),
        label=_str(assessment.get("label")),
        support=best,
        gate_status=gate_status,
        fires=fires,
        partial=partial,
        connected=bool(shown),
        onset=min(onset_turns) if fires and onset_turns else None,
        step_turns=tuple(sorted(step_turns)),
        gates=gates,
    )


def parse_judge(run: str, file: Path, line: int, offset: int, raw: bytes) -> JudgeRow:
    """A judge row. Raises :class:`Unreadable` when ``transcript_id`` is missing."""
    rec = _record(raw)
    transcript_id = _required_str(rec, "transcript_id")
    cm = dict_of(rec.get("coralmind_run"))
    meta = dict_of(cm.get("run_meta"))
    pin = dict_of(meta.get("kl"))
    stage1 = dict_of(cm.get("stage1"))
    stage2 = dict_of(cm.get("stage2"))
    observations = dicts_of(stage1.get("observations"))
    stage1_turns = {_int(t) for o in observations for t in o.get("turns") or () if _int(t) >= 1}
    failed_behaviors = frozenset(
        b
        for call in dicts_of(stage1.get("calls"))
        if call.get("status") == "judge_error"
        for b in call.get("behavior_ids") or ()
        if isinstance(b, str)
    )
    routes: dict[str, RouteResult] = {}
    for assessment in dicts_of(stage2.get("assessments")):
        result = route_result(assessment)
        if result.pathway_id:
            routes[result.pathway_id] = result
    failed = frozenset(
        p
        for call in dicts_of(stage2.get("calls"))
        if call.get("status") == "judge_error"
        for p in call.get("pathway_ids") or ()
        if isinstance(p, str) and p not in routes
    )
    tokens = dict_of(meta.get("tokens"))
    counts = dict_of(meta.get("counts"))
    status = _str(rec.get("status"), "success")
    run_status = _str(cm.get("run_status")) if cm else ""
    detail = _str(cm.get("run_error_detail")) or None
    return JudgeRow(
        run=run,
        file=file,
        line=line,
        offset=offset,
        transcript_id=transcript_id,
        evaluated_at=_str(rec.get("evaluated_at")),
        sut_model=_str(rec.get("evaluated_model_id")),
        judge_model=_str(meta.get("model")) or _str(rec.get("judge_model_id")),
        prompt_version=_str(meta.get("prompt_version")),
        prompt_hash=_str(meta.get("prompt_hash")),
        kl_digest=_str(pin.get("digest")),
        kl_version=_str(pin.get("version")),
        ok=status == "success" and run_status == "success",
        partial=_int(counts.get("call_failures")) > 0 or "partial" in (detail or ""),
        routes=routes,
        skipped=frozenset(
            _str(s.get("pathway_id"))
            for s in dicts_of(stage2.get("skipped"))
            if s.get("pathway_id")
        ),
        failed=failed,
        failed_behaviors=failed_behaviors,
        observed=frozenset(
            _str(o.get("behavior_id")) for o in observations if o.get("behavior_id")
        ),
        stage1_turns=tuple(sorted(stage1_turns)),
        tokens=(
            _int(tokens.get("prompt")),
            _int(tokens.get("completion")),
            _int(tokens.get("cached")),
        ),
        elapsed_ms=_int(meta.get("elapsed_ms")),
        call_failures=_int(counts.get("call_failures")),
        retries=_int(counts.get("retries")),
        mode=_str(meta.get("mode")) or "live",
    )


@dataclass(frozen=True)
class FileRows:
    """What one results file holds, cached by ``(size, mtime)``."""

    sims: tuple[SimRow, ...] = ()
    judges: tuple[JudgeRow, ...] = ()
    problems: tuple[ReadProblem, ...] = ()
    checks: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    """``(transcript_id, verdict)`` for each fidelity check record that has a verdict."""


def read_check_file(path: Path) -> FileRows:
    checks: list[tuple[str, Mapping[str, Any]]] = []
    problems: list[ReadProblem] = []
    for line, _, raw in _lines(path.read_bytes()):
        try:
            rec = _record(raw)
            transcript_id = _required_str(rec, "transcript_id")
        except Unreadable as exc:
            problems.append(ReadProblem(file=path, line=line, error=str(exc)))
            continue
        verdict = rec.get("fidelity")
        if isinstance(verdict, dict):
            checks.append((transcript_id, verdict))
    return FileRows(checks=tuple(checks), problems=tuple(problems))


def read_sim_file(run: str, batch: str, path: Path) -> FileRows:
    rows: list[SimRow] = []
    for line, offset, raw in _lines(path.read_bytes()):
        try:
            rows.append(parse_sim(run, batch, path, line, offset, raw))
        except Unreadable as exc:
            rows.append(
                SimRow(
                    run=run,
                    batch=batch,
                    file=path,
                    line=line,
                    offset=offset,
                    transcript_id=f"unreadable:{run}/{path.name}:{line}",
                    error=str(exc),
                )
            )
    return FileRows(sims=tuple(rows))


def read_judge_file(run: str, path: Path) -> FileRows:
    rows: list[JudgeRow] = []
    problems: list[ReadProblem] = []
    for line, offset, raw in _lines(path.read_bytes()):
        try:
            rows.append(parse_judge(run, path, line, offset, raw))
        except Unreadable as exc:
            problems.append(ReadProblem(file=path, line=line, error=str(exc)))
    return FileRows(judges=tuple(rows), problems=tuple(problems))


def read_line(path: Path, offset: int, transcript_id: str | None = None) -> dict[str, Any]:
    """The record at ``offset``: one transcript or judge line, read without the rest.

    With ``transcript_id``, raises :class:`StaleRecord` when the line holds another record,
    which happens when the file was rewritten after the snapshot read it.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.readline()
    try:
        rec = _record(raw)
    except Unreadable as exc:
        if transcript_id is None:
            raise
        raise StaleRecord(f"{path.name} changed since the results were read") from exc
    if transcript_id is not None and rec.get("transcript_id") != transcript_id:
        raise StaleRecord(f"{path.name} changed since the results were read")
    return rec


# ---------------------------------------------------------------------------
# Runs, cases and personas
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


@dataclass(frozen=True)
class Persona:
    """The facts a person record holds that the viewer shows and filters on."""

    description: str
    elevated: tuple[str, ...]
    traits: Mapping[str, Any]
    clinical_context: str = ""
    behavioral_description: str = ""


@dataclass(frozen=True)
class Case:
    """A test case as the viewer reads it from a run's ``inputs/`` copy."""

    test_case_id: str
    pathway_id: str
    level: int | None
    persona_ref: str
    counterpart_of: str | None
    situation: tuple[tuple[str, str], ...]
    design: tuple[str, int] | None
    display_name: str
    record: Mapping[str, Any]
    persona: Persona | None = None
    persona_error: str | None = None

    @property
    def is_counterfactual(self) -> bool:
        return self.counterpart_of is not None


def parse_case(rec: Mapping[str, Any]) -> Case:
    """A case row. Raises :class:`Unreadable` without ``test_case_id`` or ``pathway_id``."""
    situation = dict_of(rec.get("situation"))
    provenance = dict_of(rec.get("provenance"))
    design = provenance.get("design") if isinstance(provenance.get("design"), dict) else None
    level = rec.get("evaluation_level")
    return Case(
        test_case_id=_required_str(rec, "test_case_id"),
        pathway_id=_required_str(rec, "pathway_id"),
        level=level if isinstance(level, int) else None,
        persona_ref=_str(rec.get("persona_ref")),
        counterpart_of=_str(rec.get("counterpart_of")) or None,
        situation=tuple(sorted((str(k), str(v)) for k, v in situation.items())),
        design=(_str(design.get("table")), _int(design.get("row"))) if design else None,
        display_name=_str(rec.get("display_name")),
        record=rec,
    )


_PERSONA_ID = re.compile(r"p-[a-z0-9]+(?:-[a-z0-9]+)*")


def load_persona(personas_dir: Path | None, ref: str) -> Persona | str:
    """The persona ``ref`` names, or why it cannot be read."""
    if not ref:
        return "no persona_ref"
    persona_id, _, version = ref.partition("@")
    if personas_dir is None or not _PERSONA_ID.fullmatch(persona_id):
        return f"persona record not found: {ref}"
    path = personas_dir / f"{persona_id}.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"persona record not found: {ref}"
    if not isinstance(rec, dict) or (version and str(rec.get("version")) != version):
        return f"persona record not found: {ref}"
    rendering = dict_of(rec.get("rendering"))
    traits = dict_of(rendering.get("traits"))
    return Persona(
        description=_str(rendering.get("persona_description")),
        elevated=tuple(
            _str(e.get("value")) for e in dicts_of(rec.get("elevated")) if e.get("value")
        ),
        traits=traits,
        clinical_context=_str(rendering.get("clinical_context")),
        behavioral_description=_str(rendering.get("behavioral_description")),
    )


@dataclass(frozen=True)
class RunInfo:
    """One results directory, with the inputs copied into it."""

    name: str
    directory: Path
    batches: tuple[Batch, ...]
    manifest: Mapping[str, Any]
    manifest_path: Path | None
    cases_file: Path | None
    suite_id: str | None
    """The version of the test case file the run used: ``<suite name>@<content hash>``."""
    suite_name: str | None
    """The test case file's path under ``test_case_records/``, without ``.jsonl``."""
    fact_mode: str
    level: int | None
    sut_models: Mapping[str, str]
    personas_dir: Path | None
    created: float = 0.0
    """When the run was made: the time in its directory name, else its newest sim file."""

    @property
    def run_id(self) -> str:
        return _str(self.manifest.get("run_id"))


RECORDS_DIR = "test_case_records"


def file_digest(path: Path) -> str:
    """The first 12 hex digits of a file's SHA-256: the version of a test case file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def suite_name_of(records_path: str, cases_file: Path) -> str:
    """A suite's name: the manifest's test case path under ``test_case_records/``.

    A path outside ``test_case_records/``, or no path, names the suite by the copied file's stem.
    """
    parts = Path(records_path).with_suffix("").parts if records_path else ()
    if RECORDS_DIR not in parts:
        return cases_file.stem
    return "/".join(parts[parts.index(RECORDS_DIR) + 1 :])


def records_files() -> dict[str, Path]:
    """Every test case file in the data repository's ``test_case_records/``, by suite name."""
    root = Path(resolve_data_path(RECORDS_DIR))
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).with_suffix("").as_posix(): path
        for path in sorted(root.rglob("*.jsonl"))
        if path.is_file()
    }


def _manifest(inputs: Path) -> tuple[dict[str, Any], Path | None]:
    for path in sorted(inputs.glob("*")):
        if path.suffix not in (".yaml", ".yml", ".json") or not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and "system_under_test" in data:
            return data, path
    return {}, None


def fact_mode_of(manifest: Mapping[str, Any]) -> str:
    """The run's fact mode: the one mode its ``fact_modes`` names, else 'not recorded'."""
    modes = manifest.get("fact_modes") or manifest.get("fact_mode")
    if isinstance(modes, str):
        return modes
    values: list[str] = []
    if isinstance(modes, dict):
        values = [str(v) for v in modes.values()]
    elif isinstance(modes, list):
        values = [str(v) for v in modes]
    distinct = sorted(set(values))
    return " + ".join(distinct) if distinct else FACT_MODE_UNRECORDED


def _sut_models(directory: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    models: dict[str, str] = {}
    for sut in dicts_of(manifest.get("system_under_test")):
        model_id = _str(sut.get("model_id"))
        label = _str(sut.get("label"))
        if model_id:
            models[label or _slug(model_id)] = label or model_id
    for index in sorted(directory.glob("*.index.json")):
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in dicts_of(data.get("models") if isinstance(data, dict) else None):
            key, model_id = _str(entry.get("output_key")), _str(entry.get("model_id"))
            if key and model_id:
                models.setdefault(key, model_id)
    return models


def checker_model(manifest: Mapping[str, Any]) -> str:
    """The fidelity checker's model: its own, else the person simulator's, which it reuses."""
    simulation = dict_of(manifest.get("simulation"))
    fidelity = dict_of(simulation.get("elicitation_fidelity"))
    own = _str(dict_of(fidelity.get("model")).get("model_id"))
    return own or _str(dict_of(simulation.get("person_response_model")).get("model_id"))


def _personas_dir(inputs: Path, manifest: Mapping[str, Any]) -> Path:
    """The run's own copy of its personas when it has one, else the data repository's."""
    own = inputs / "persona_records"
    if own.is_dir():
        return own
    return Path(resolve_data_path(_str(manifest.get("persona_records_path"), "persona_records")))


def run_info(run_name: str, directory: Path, batches: tuple[Batch, ...]) -> RunInfo:
    inputs = directory / "inputs"
    manifest: dict[str, Any] = {}
    manifest_path: Path | None = None
    cases_file: Path | None = None
    if inputs.is_dir():
        manifest, manifest_path = _manifest(inputs)
        candidates = sorted(p for p in inputs.glob("*.jsonl") if p.is_file())
        wanted = Path(_str(manifest.get("test_case_records_path"))).name
        named = [p for p in candidates if p.name == wanted]
        found = named or candidates
        cases_file = found[0] if found else None
    suite_id = None
    suite_name = None
    if cases_file is not None:
        suite_name = suite_name_of(_str(manifest.get("test_case_records_path")), cases_file)
        suite_id = f"{suite_name}@{file_digest(cases_file)}"
    level = manifest.get("evaluation_level")
    return RunInfo(
        name=run_name,
        directory=directory,
        batches=batches,
        manifest=manifest,
        manifest_path=manifest_path,
        cases_file=cases_file,
        suite_id=suite_id,
        suite_name=suite_name,
        fact_mode=fact_mode_of(manifest),
        level=level if isinstance(level, int) else None,
        sut_models=_sut_models(directory, manifest),
        personas_dir=_personas_dir(inputs, manifest),
        created=_created(run_name, batches),
    )


_RUN_STAMP = re.compile(r"-(\d{8}-\d{6})(?:-\d+)?$")


def _created(run_name: str, batches: tuple[Batch, ...]) -> float:
    """The ``-YYYYMMDD-HHMMSS`` the CLI appends to a run directory, else the newest sim file.

    Judging writes into the run's directory later, so judge files never date a run.
    """
    stamp = _RUN_STAMP.search(run_name)
    if stamp is not None:
        return datetime.strptime(stamp.group(1), "%Y%m%d-%H%M%S").timestamp()
    return max((p.stat().st_mtime for b in batches for p in b.sim_files), default=0.0)


# ---------------------------------------------------------------------------
# Knowledge Library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """One composition part of a route, from its KL entry."""

    kind: str
    id: str
    role: str

    @property
    def is_condition(self) -> bool:
        return self.role == "condition"


@dataclass(frozen=True)
class ImpactEdge:
    id: str
    sign: str


@dataclass(frozen=True)
class RouteDef:
    """A route's structure, read from its KL entry."""

    id: str
    did: str
    label: str
    definition: str
    parts: tuple[Part, ...]
    composition_instruction: str
    neighbors: tuple[str, ...]
    impacts: tuple[ImpactEdge, ...]

    @property
    def observations(self) -> tuple[Part, ...]:
        return tuple(p for p in self.parts if not p.is_condition)

    @property
    def conditions(self) -> tuple[Part, ...]:
        return tuple(p for p in self.parts if p.is_condition)


@dataclass(frozen=True)
class EntryName:
    did: str
    label: str
    type: str
    definition: str
    dimension: str = ""


@dataclass(frozen=True)
class Library:
    """The KL entries the viewer reads: route structure, and every entry's name."""

    version: str
    digest: str
    routes: Mapping[str, RouteDef]
    names: Mapping[str, EntryName]
    criteria_parts: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """A user behavior's lettered criteria, ``{letter: clause}``."""
    parts_needed: Mapping[str, int] = field(default_factory=dict)
    """How many distinct lettered parts a user behavior needs; 1 when any one counts."""

    def name(self, entry_id: str) -> EntryName | None:
        return self.names.get(entry_id)


def _value(item: Any) -> str:
    return str(getattr(item, "value", item))


def load_library(path: str | Path | None = None) -> Library:
    """Load the KL through ``corallab.kl.load_library``. Raises whatever the load raises."""
    from corallab.kl import load_library as load

    kl = load(path)
    names: dict[str, EntryName] = {}
    for entry in kl.all_entries():
        names[entry.id] = EntryName(
            did=entry.display_id,
            label=entry.label,
            type=_value(entry.type),
            definition=_str(getattr(entry, "definition", "")),
            dimension=_str(getattr(getattr(entry, "body", None), "dimension", "")),
        )
    routes: dict[str, RouteDef] = {}
    for pathway in kl.pathways:
        body = pathway.body
        routes[pathway.id] = RouteDef(
            id=pathway.id,
            did=pathway.display_id,
            label=pathway.label,
            definition=pathway.definition,
            parts=tuple(
                Part(kind=_value(p.kind), id=p.id, role=_value(p.role))
                for p in body.composition_parts
            ),
            composition_instruction=body.composition_instruction or "",
            neighbors=tuple(d.neighbor for d in pathway.distinguish_from),
            impacts=tuple(ImpactEdge(e.impact, _value(e.sign)) for e in body.impacts),
        )
    from corallab.judging.judges.kl_two_stage import criteria_elements, required_elements

    criteria_parts = {}
    parts_needed = {}
    for entry in kl.all_entries():
        detail = getattr(entry, "body", None)
        criteria = getattr(detail, "observable_criteria", None) or getattr(
            detail, "absence_criteria", None
        )
        if _value(entry.type) == "user_behavior" and criteria:
            criteria_parts[entry.id] = criteria_elements(criteria)
            parts_needed[entry.id] = required_elements(criteria)
    return Library(
        version=kl.version,
        digest=kl.digest,
        routes=routes,
        names=names,
        criteria_parts=criteria_parts,
        parts_needed=parts_needed,
    )


# ---------------------------------------------------------------------------
# Report file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgePass:
    """Judge records sharing a judge model, prompt and KL digest: one "judge setup"."""

    key: str
    id: str
    judge_model: str
    prompt_version: str
    prompt_hash: str
    kl_digest: str
    kl_version: str
    transcripts: int
    latest: str


@dataclass(frozen=True)
class Suite:
    """Result directories whose copied test-case file has the same content hash."""

    id: str
    stem: str
    hash: str
    runs: tuple[str, ...]
    cases: Mapping[str, Case]
    case_problems: tuple[str, ...]
    passes: tuple[JudgePass, ...]
    updated: float


@dataclass(frozen=True)
class SuiteFile:
    """A test case file and every run made from it, across the file's versions."""

    name: str
    path: Path | None
    """The file in ``test_case_records/``, when the data repository has it."""
    source: Path | None
    """Where ``cases`` were read: ``path``, else the newest run's copy."""
    digest: str | None
    """The version of ``source``."""
    cases: Mapping[str, Case]
    case_problems: tuple[str, ...]
    versions: tuple[str, ...]
    """Suite ids of the versions runs used, newest run first."""
    runs: tuple[str, ...]
    """Run names, newest first."""


@dataclass(frozen=True)
class Snapshot:
    """One consistent reading of the results directory. Never mutated after it is built."""

    root: Path
    built_at: datetime
    runs: Mapping[str, RunInfo]
    sims: tuple[SimRow, ...]
    judges: tuple[JudgeRow, ...]
    judge_problems: tuple[ReadProblem, ...]
    suites: Mapping[str, Suite]
    library: Library | None
    library_error: str | None
    signature: tuple[tuple[str, int, int], ...]
    suite_files: Mapping[str, SuiteFile] = field(default_factory=dict)
    sims_by_key: Mapping[tuple[str, str], SimRow] = field(default_factory=dict)
    judges_by_key: Mapping[tuple[str, str], tuple[JudgeRow, ...]] = field(default_factory=dict)
    check_verdicts: Mapping[tuple[str, str], Mapping[str, Any]] = field(default_factory=dict)
    """The fidelity check's verdict on each transcript, by ``SimRow.key``, from its check file."""

    def sut_model(self, sim: SimRow) -> str:
        """The system tested, named by its manifest label or model id.

        A line with no readable system goes to the run's only system when it has one, and
        otherwise to :data:`UNKNOWN_SUT`, which no system column shows.
        """
        run = self.runs.get(sim.run)
        model = run.sut_models.get(sim.sut_key) if run else None
        if model:
            return model
        for judge in self.judges_by_key.get(sim.key, ()):
            if judge.sut_model:
                return judge.sut_model
        if run is not None and len(set(run.sut_models.values())) == 1:
            return next(iter(run.sut_models.values()))
        return sim.sut_key or UNKNOWN_SUT


def pass_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def passes_of(judges: Iterable[JudgeRow]) -> tuple[JudgePass, ...]:
    """Judge records grouped into judge passes, the one with the most transcripts first."""
    grouped: dict[str, list[JudgeRow]] = defaultdict(list)
    for judge in judges:
        if judge.has_meta:
            grouped[judge.pass_key].append(judge)
    passes = [
        JudgePass(
            key=key,
            id=pass_id(key),
            judge_model=rows[0].judge_model,
            prompt_version=rows[0].prompt_version,
            prompt_hash=rows[0].prompt_hash,
            kl_digest=rows[0].kl_digest,
            kl_version=rows[0].kl_version,
            transcripts=len({r.key for r in rows}),
            latest=max(r.evaluated_at for r in rows),
        )
        for key, rows in grouped.items()
    ]
    passes.sort(key=lambda p: (p.transcripts, p.latest), reverse=True)
    return tuple(passes)


def _cases(path: Path, personas_dir: Path | None) -> tuple[dict[str, Case], list[str]]:
    cases: dict[str, Case] = {}
    problems: list[str] = []
    for line, _, raw in _lines(path.read_bytes()):
        try:
            case = parse_case(_record(raw))
        except Unreadable as exc:
            problems.append(f"{path.name}:{line}: {exc}")
            continue
        persona = load_persona(personas_dir, case.persona_ref)
        if isinstance(persona, Persona):
            case = replace(case, persona=persona)
        else:
            case = replace(case, persona_error=persona)
        cases[case.test_case_id] = case
    return cases, problems


def _signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """``(path, size, mtime)`` of every file the snapshot reads."""
    items: list[tuple[str, int, int]] = []
    for path in records_files().values():
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append((str(path), stat.st_size, stat.st_mtime_ns))
    if not root.is_dir():
        return tuple(items)
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for pattern in (
            "*-sim.jsonl",
            "*-judge.jsonl",
            "*-check.jsonl",
            "*.index.json",
            "batch-*/*-sim.jsonl",
            "batch-*/*-judge.jsonl",
            "batch-*/*-check.jsonl",
            "inputs/*",
        ):
            for path in sorted(run_dir.glob(pattern)):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                items.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(items)


class FileReadError(Exception):
    """A results file that could not be read. Names the file."""

    def __init__(self, path: Path, error: Exception) -> None:
        self.path = path
        self.error = error
        super().__init__(f"{error} ({path})")


class _FileCache:
    """Rows per results file, keyed by ``(size, mtime)``. A changed file is read again."""

    def __init__(self) -> None:
        self._rows: dict[Path, tuple[tuple[int, int], FileRows]] = {}
        self.reads = 0

    def get(self, path: Path, read: Callable[[], FileRows]) -> FileRows:
        try:
            stat = path.stat()
        except OSError as exc:
            raise FileReadError(path, exc) from exc
        key = (stat.st_size, stat.st_mtime_ns)
        cached = self._rows.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            rows = read()
        except OSError as exc:
            raise FileReadError(path, exc) from exc
        self.reads += 1
        self._rows[path] = (key, rows)
        return rows


def build_snapshot(
    root: Path,
    *,
    cache: _FileCache | None = None,
    library: Library | None = None,
    library_error: str | None = None,
) -> Snapshot:
    """Read ``root`` into a snapshot. Raises :class:`FileReadError` for a file that won't open.

    Files read before the failure stay in ``cache``, so a retry reads only what is left.
    """
    cache = cache or _FileCache()
    signature = _signature(root)
    runs: dict[str, RunInfo] = {}
    jobs: list[tuple[Path, Callable[[], FileRows]]] = []
    check_jobs: list[tuple[str, Path]] = []
    for run in scan(root):
        info = run_info(run.name, run.directory, run.batches)
        runs[run.name] = info
        for batch in run.batches:
            jobs.extend(
                (p, partial(read_sim_file, run.name, batch.name, p)) for p in batch.sim_files
            )
            jobs.extend((p, partial(read_judge_file, run.name, p)) for p in batch.judge_files)
            check_jobs.extend((run.name, p) for p in batch.check_files)
    sims: list[SimRow] = []
    judges: list[JudgeRow] = []
    problems: list[ReadProblem] = []
    for path, read in jobs:
        rows = cache.get(path, read)
        sims.extend(rows.sims)
        judges.extend(rows.judges)
        problems.extend(rows.problems)
    verdicts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for run_name, path in check_jobs:
        rows = cache.get(path, partial(read_check_file, path))
        verdicts.update({(run_name, tid): v for tid, v in rows.checks})
        problems.extend(rows.problems)
    sims = [with_check(s, verdicts.get(s.key)) for s in sims]

    judges_by_key: dict[tuple[str, str], list[JudgeRow]] = defaultdict(list)
    for judge in judges:
        judges_by_key[judge.key].append(judge)
    suites = _suites(runs, sims, judges_by_key)
    suite_files = _suite_files(runs)
    return Snapshot(
        root=root,
        built_at=datetime.now(UTC),
        runs=runs,
        sims=tuple(sims),
        judges=tuple(judges),
        judge_problems=tuple(problems),
        suites=suites,
        library=library,
        library_error=library_error,
        signature=signature,
        sims_by_key={sim.key: sim for sim in sims},
        judges_by_key={k: tuple(v) for k, v in judges_by_key.items()},
        suite_files=suite_files,
        check_verdicts=verdicts,
    )


def _suite_files(runs: Mapping[str, RunInfo]) -> dict[str, SuiteFile]:
    files = records_files()
    by_name: dict[str, list[RunInfo]] = defaultdict(list)
    for info in runs.values():
        if info.suite_name is not None:
            by_name[info.suite_name].append(info)
    personas = Path(resolve_data_path("persona_records"))
    out: dict[str, SuiteFile] = {}
    for name in sorted(set(files) | set(by_name)):
        infos = sorted(by_name.get(name, []), key=lambda r: (r.created, r.name), reverse=True)
        path = files.get(name)
        source = path or (infos[0].cases_file if infos else None)
        cases: Mapping[str, Case] = {}
        problems: list[str] = []
        if source is not None:
            cases, problems = _cases(source, personas if path else infos[0].personas_dir)
        out[name] = SuiteFile(
            name=name,
            path=path,
            source=source,
            digest=file_digest(source) if source is not None else None,
            cases=cases,
            case_problems=tuple(problems),
            versions=tuple(dict.fromkeys(r.suite_id for r in infos if r.suite_id)),
            runs=tuple(r.name for r in infos),
        )
    return out


def _suites(
    runs: Mapping[str, RunInfo],
    sims: Iterable[SimRow],
    judges_by_key: Mapping[tuple[str, str], list[JudgeRow]],
) -> dict[str, Suite]:
    members: dict[str, list[RunInfo]] = defaultdict(list)
    for info in runs.values():
        if info.suite_id is not None:
            members[info.suite_id].append(info)
    sims_by_run: dict[str, list[SimRow]] = defaultdict(list)
    for sim in sims:
        sims_by_run[sim.run].append(sim)
    suites: dict[str, Suite] = {}
    for suite_id, infos in sorted(members.items()):
        first = infos[0]
        assert first.cases_file is not None
        cases, problems = _cases(first.cases_file, first.personas_dir)
        suite_judges = [
            judge
            for info in infos
            for sim in sims_by_run[info.name]
            for judge in judges_by_key.get(sim.key, ())
        ]
        stem, _, digest = suite_id.partition("@")
        updated = max(info.created for info in infos)
        suites[suite_id] = Suite(
            id=suite_id,
            stem=stem,
            hash=digest,
            runs=tuple(sorted(info.name for info in infos)),
            cases=cases,
            case_problems=tuple(problems),
            passes=passes_of(suite_judges),
            updated=updated,
        )
    return suites


# ---------------------------------------------------------------------------
# Index: the current snapshot, rebuilt when a file changes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexState:
    """The latest snapshot, or why there is none, and why the last rebuild failed if it did."""

    snapshot: Snapshot | None
    first_error: str | None
    stale_since: datetime | None
    stale_error: str | None
    root: Path


class Index:
    """The latest snapshot of a results directory, rebuilt on request.

    A rebuild reads only the files that changed. A failed rebuild keeps the previous snapshot
    and records why it is stale. The Knowledge Library is loaded once, at the first rebuild.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache = _FileCache()
        self._snapshot: Snapshot | None = None
        self._first_error: str | None = None
        self._stale_since: datetime | None = None
        self._stale_error: str | None = None
        self._library: Library | None = None
        self._library_error: str | None = None
        self._library_loaded = False

    @property
    def file_reads(self) -> int:
        return self._cache.reads

    def status(self) -> IndexState:
        return IndexState(
            snapshot=self._snapshot,
            first_error=self._first_error,
            stale_since=self._stale_since,
            stale_error=self._stale_error,
            root=self.root,
        )

    def rebuild(self) -> Snapshot | None:
        """Read the results again. Returns the current snapshot; a failure leaves it unchanged."""
        self._ensure_library()
        try:
            snapshot = build_snapshot(
                self.root,
                cache=self._cache,
                library=self._library,
                library_error=self._library_error,
            )
        except FileReadError as exc:
            message = f"{exc.error} ({exc.path})"
            logger.warning("Could not read results: %s", message)
            self._record_failure(message)
            return self._snapshot
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("Could not read results")
            self._record_failure(message)
            return self._snapshot
        self._snapshot = snapshot
        self._first_error = None
        self._stale_since = None
        self._stale_error = None
        return snapshot

    def _record_failure(self, message: str) -> None:
        if self._snapshot is None:
            self._first_error = message
        else:
            self._stale_since = self._stale_since or datetime.now()
            self._stale_error = message

    def _ensure_library(self) -> None:
        if self._library_loaded:
            return
        self._library_loaded = True
        try:
            self._library = load_library()
        except Exception as exc:
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            self._library_error = first
            logger.warning("Knowledge Library unavailable: %s", first)


# ---------------------------------------------------------------------------
# The transcripts and routes tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transcript:
    """A row of the ``transcripts`` table."""

    sim: SimRow
    run: str
    suite_id: str | None
    case: Case | None
    sut: str
    level: int | None
    fact_mode: str
    bucket: str
    judge: JudgeRow | None

    @property
    def id(self) -> str:
        return self.sim.transcript_id

    @property
    def key(self) -> tuple[str, str]:
        return self.sim.key

    @property
    def test_case_id(self) -> str:
        return self.sim.test_case_id

    @property
    def target(self) -> str:
        return self.case.pathway_id if self.case else ""


@dataclass(frozen=True, slots=True)
class Route:
    """A row of the ``routes`` table: one judged transcript and one route of its suite.

    ``assessment`` is how Stage 2 treated the route (``assessed``, ``skipped``, ``failed``
    or ``missing``); ``outcome`` is what the rate counts: ``fires``, ``partial`` or ``not``,
    or a reason it is left out (``gate_unknown``, ``failed``, ``kl_mismatch``).
    """

    transcript: Transcript
    route: str
    is_target: bool
    assessment: str
    outcome: str
    result: RouteResult | None
    neighbors_fired: tuple[str, ...]


def bucket_of(sim: SimRow, judge: JudgeRow | None, level: int | None) -> str:
    """The outcome bucket for one transcript, given the judge record selected for it."""
    if sim.error is not None:
        return "unreadable"
    if sim.sim_failure is not None:
        return "sim_failure"
    if level in (1, 2):
        return "judged" if judge is not None and judge.ok else "not_judged"
    if sim.copied_turn is not None:
        return "invalid"
    if sim.fidelity is None:
        return "unchecked"
    if sim.fidelity == "invalid":
        return "invalid"
    if sim.fidelity == "not_tested":
        return "not_tested"
    if sim.fidelity != "valid":
        return "unchecked"
    return "judged" if judge is not None and judge.ok else "not_judged"


def _latest(judges: Iterable[JudgeRow]) -> JudgeRow | None:
    return max(judges, key=lambda j: j.evaluated_at, default=None)


def route_row(
    transcript: Transcript,
    route: str,
    library: Library | None,
    firing: frozenset[str] | None = None,
) -> Route:
    """One ``routes`` row. ``transcript`` must be judged; ``firing`` is its judge's firing set."""
    judge = transcript.judge
    assert judge is not None
    is_target = route == transcript.target
    result = judge.routes.get(route)
    if library is not None and judge.kl_digest and judge.kl_digest != library.digest:
        return Route(
            transcript,
            route,
            is_target,
            "assessed" if result else "missing",
            "kl_mismatch",
            result,
            (),
        )
    if result is not None:
        assessment, outcome = "assessed", result.outcome
    elif route in judge.failed:
        assessment, outcome = "failed", "failed"
    elif route in judge.skipped:
        if judge.partial and _skip_may_have_failed(judge, route, library):
            assessment, outcome = "failed", "failed"
        else:
            assessment, outcome = "skipped", "not"
    else:
        assessment, outcome = "missing", "failed"
    neighbors: tuple[str, ...] = ()
    if library is not None and route in library.routes:
        firing = judge.firing if firing is None else firing
        neighbors = tuple(n for n in library.routes[route].neighbors if n in firing)
    return Route(transcript, route, is_target, assessment, outcome, result, neighbors)


def _skip_may_have_failed(judge: JudgeRow, route: str, library: Library | None) -> bool:
    """A skipped route whose participants a failed Stage 1 group was to detect."""
    if not judge.failed_behaviors:
        return False
    if library is None or route not in library.routes:
        return True
    return any(p.id in judge.failed_behaviors for p in library.routes[route].observations)


@dataclass(frozen=True)
class RunFacts:
    """One run's transcripts. Each is joined to its latest judge record in the run."""

    run: RunInfo
    transcripts: tuple[Transcript, ...]
    judge_problems: tuple[ReadProblem, ...]
    orphans: int
    library: Library | None

    def routes(self, transcript: Transcript) -> tuple[Route, ...]:
        """Every route the judge assessed, the target first."""
        if transcript.bucket != "judged" or transcript.judge is None:
            return ()
        ids = [transcript.target] if transcript.target else []
        ids += sorted(r for r in transcript.judge.routes if r != transcript.target)
        return tuple(route_row(transcript, r, self.library) for r in ids)


def run_facts(snapshot: Snapshot, run_name: str) -> RunFacts:
    info = snapshot.runs[run_name]
    suite = snapshot.suites.get(info.suite_id) if info.suite_id else None
    cases: Mapping[str, Case] = suite.cases if suite else {}
    if suite is None and info.cases_file is not None:
        cases = _cases(info.cases_file, info.personas_dir)[0]
    transcripts: list[Transcript] = []
    for sim in snapshot.sims:
        if sim.run != run_name:
            continue
        judge = _latest(snapshot.judges_by_key.get(sim.key, ()))
        case = cases.get(sim.test_case_id)
        level = case.level if case and case.level else info.level
        transcripts.append(
            Transcript(
                sim=sim,
                run=run_name,
                suite_id=info.suite_id,
                case=case,
                sut=snapshot.sut_model(sim),
                level=level,
                fact_mode=info.fact_mode,
                bucket=bucket_of(sim, judge, level),
                judge=judge,
            )
        )
    orphans = sum(
        1
        for judge in snapshot.judges
        if judge.run == run_name and judge.key not in snapshot.sims_by_key
    )
    problems = tuple(p for p in snapshot.judge_problems if p.file.is_relative_to(info.directory))
    return RunFacts(
        run=info,
        transcripts=tuple(transcripts),
        judge_problems=problems,
        orphans=orphans,
        library=snapshot.library,
    )
