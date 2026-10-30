"""Write a snapshot as the page data the Next app reads: one JSON file per page model.

Each file is written under a temporary name and renamed into place, and a file whose content
has not changed is left untouched, so a dev server never reads a half-written file and a
re-ingest rewrites only what changed. Every derived fact is worked out here; the app only
selects, counts and lays out.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from corallab.paths import data_dir
from corallab.simulation.common.copied_messages import copied_ai_message
from coralmind_viewer import export_models as M
from coralmind_viewer import facts as F
from coralmind_viewer.facts import dict_of, dicts_of
from coralmind_viewer.trace_loaders import (
    JudgeCall,
    check_call_from_rows,
    check_trace_path,
    judge_calls,
    judge_trace_path,
    sim_driver_from_rows,
    sim_trace_path,
    trace_row,
    trace_rows,
)


def local_time(timestamp: float) -> str:
    """A time in seconds as ISO local time without a zone, as the run's own stamp reads."""
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds") if timestamp else ""


def key_of(*parts: str) -> str:
    """The file key for a suite, run or transcript: stable, and safe in any file system."""
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@dataclass
class Writer:
    """Writes generated files under ``out``, skipping unchanged ones, and lists what it wrote."""

    out: Path
    written: set[Path] = field(default_factory=set)
    changed: int = 0

    def write(self, relative: str, model: BaseModel) -> None:
        path = self.out / relative
        data = model.model_dump_json(exclude_none=False).encode("utf-8")
        self.written.add(path)
        try:
            if path.read_bytes() == data:
                return
        except OSError:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self.changed += 1

    def prune(self) -> int:
        """Remove generated JSON this run did not write, such as a deleted run's pages."""
        removed = 0
        for path in self.out.rglob("*.json"):
            if path not in self.written:
                path.unlink()
                removed += 1
        return removed


def shown_suite_files(snapshot: F.Snapshot) -> list[F.SuiteFile]:
    """The test case files that runs in the results used."""
    return [f for f in snapshot.suite_files.values() if f.runs]


def export(snapshot: F.Snapshot, out: Path) -> Writer:
    """Write every page model for ``snapshot`` under ``out``. Returns the writer's counts.

    Transcript pages are written first and the lists that link to them last, so an export
    that fails part way leaves every list pointing at the previous, complete data.
    """
    writer = Writer(out)
    ctx = _Context(snapshot, data_dir())
    lists: list[tuple[str, BaseModel]] = [("library.json", library_data(snapshot.library))]
    lists += [
        (f"suite-files/{key_of(f.name)}.json", suite_file_data(ctx, f.name))
        for f in shown_suite_files(snapshot)
    ]
    lists += [(f"runs/{key_of(r)}.json", run_data(ctx, r)) for r in snapshot.runs]
    places: dict[str, list[M.TranscriptPlace]] = defaultdict(list)
    for sim in sorted(snapshot.sims, key=lambda s: (snapshot.runs[s.run].created, s.run, s.line)):
        case_key = key_of(sim.run, sim.transcript_id)
        writer.write(f"cases/{case_key}.json", case_data(ctx, sim))
        writer.write(f"records/{case_key}.json", records_data(ctx, sim))
        if sim.error is None:
            places[sim.transcript_id].append(M.TranscriptPlace(run=sim.run, key=case_key))
    lists.append(("transcripts.json", M.TranscriptsData(places=dict(places))))
    lists.append(("index.json", index_data(ctx)))
    for relative, model in lists:
        writer.write(relative, model)
    writer.prune()
    return writer


# ---------------------------------------------------------------------------
# Shared reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleModels:
    suts: list[str]
    simulator: str
    checker: str
    judges: list[str]


class _Context:
    """The snapshot with the caches the per-transcript pages share."""

    def __init__(self, snapshot: F.Snapshot, data_root: Path) -> None:
        self.snapshot = snapshot
        self.data_root = data_root
        self.library = snapshot.library
        self._run_facts: dict[str, F.RunFacts] = {}
        self._transcripts: dict[tuple[str, str], F.Transcript] = {}
        self._sim_traces: dict[Path, list[dict[str, Any]]] = {}
        self._check_traces: dict[Path, list[dict[str, Any]]] = {}
        self._suite_sims: dict[str, dict[str, list[F.SimRow]]] = {}

    def rel(self, path: Path | None) -> str | None:
        """``path`` relative to the data repository, with ``/`` separators."""
        if path is None:
            return None
        return Path(os.path.relpath(path, self.data_root)).as_posix()

    def run_facts(self, name: str) -> F.RunFacts:
        if name not in self._run_facts:
            self._run_facts[name] = F.run_facts(self.snapshot, name)
        return self._run_facts[name]

    def roles(self, name: str) -> RoleModels:
        """The model in each role of a run; a role the run did not fill is ``""`` or empty."""
        facts = self.run_facts(name)
        simulation = dict_of(facts.run.manifest.get("simulation"))
        fidelity = dict_of(simulation.get("elicitation_fidelity"))
        checks = facts.run.level == 3 and fidelity.get("enabled", True) is not False
        return RoleModels(
            suts=sorted({t.sut for t in facts.transcripts if t.bucket != "unreadable"}),
            simulator=str(dict_of(simulation.get("person_response_model")).get("model_id") or ""),
            checker=F.checker_model(facts.run.manifest) if checks else "",
            judges=sorted(
                {t.judge.judge_model for t in facts.transcripts if t.judge and t.judge.judge_model}
            ),
        )

    def transcript(self, sim: F.SimRow) -> F.Transcript:
        """The Run-mode row of ``sim``: joined to its latest judge record in the run."""
        if sim.key not in self._transcripts:
            for t in self.run_facts(sim.run).transcripts:
                self._transcripts[t.key] = t
        return self._transcripts[sim.key]

    def suite_sims(self, suite: F.Suite) -> dict[str, list[F.SimRow]]:
        """The suite's readable transcript lines, by test case id."""
        if suite.id not in self._suite_sims:
            by_case: dict[str, list[F.SimRow]] = defaultdict(list)
            for s in self.snapshot.sims:
                if s.run in suite.runs and s.error is None:
                    by_case[s.test_case_id].append(s)
            self._suite_sims[suite.id] = by_case
        return self._suite_sims[suite.id]

    def sim_trace_rows(self, sim_file: Path, test_case_id: str) -> list[dict[str, Any]]:
        """The sim trace rows that name ``test_case_id`` or no test case, read once per file."""
        path = sim_trace_path(sim_file)
        if path not in self._sim_traces:
            rows: list[dict[str, Any]] = []
            try:
                with path.open("rb") as handle:
                    for raw in handle:
                        try:
                            row = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            rows.append(row)
            except OSError:
                rows = []
            self._sim_traces[path] = rows
        return [
            r
            for r in self._sim_traces[path]
            if r.get("test_case_id") in (None, test_case_id) or "test_case_id" not in r
        ]

    def check_trace_rows(self, sim_file: Path) -> list[dict[str, Any]]:
        """Every row of the check trace beside ``sim_file``, read once per file."""
        path = check_trace_path(sim_file)
        if path not in self._check_traces:
            self._check_traces[path] = trace_rows(path)
        return self._check_traces[path]


def _problems(items: Iterable[F.ReadProblem]) -> list[M.FileProblem]:
    return [M.FileProblem(file=p.file.name, line=p.line, error=p.error) for p in items]


def _pass(item: F.JudgePass) -> M.JudgePassData:
    return M.JudgePassData(
        key=item.key,
        id=item.id,
        judge_model=item.judge_model,
        prompt_version=item.prompt_version,
        prompt_hash=item.prompt_hash,
        kl_digest=item.kl_digest,
        kl_version=item.kl_version,
        transcripts=item.transcripts,
        latest=item.latest,
    )


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def _role_definitions() -> dict[str, str]:
    from coralmap.vocabularies import PartRole

    return {str(role.value): str(role.description) for role in PartRole}


def library_data(library: F.Library | None) -> M.LibraryData:
    if library is None:
        return M.LibraryData(version="", digest="", names={}, routes={}, roles={})
    return M.LibraryData(
        version=library.version,
        digest=library.digest,
        names={
            k: M.EntryNameData(
                did=v.did,
                label=v.label,
                type=v.type,
                definition=v.definition,
                dimension=v.dimension,
            )
            for k, v in library.names.items()
        },
        routes={
            k: M.RouteDefData(
                id=r.id,
                did=r.did,
                label=r.label,
                definition=r.definition,
                parts=[M.PartData(kind=p.kind, id=p.id, role=p.role) for p in r.parts],
                composition_instruction=r.composition_instruction,
                neighbors=list(r.neighbors),
                impacts=[M.ImpactData(id=e.id, sign=e.sign) for e in r.impacts],
                direction=_direction(r),
            )
            for k, r in library.routes.items()
        },
        roles=_role_definitions(),
    )


def _direction(route: F.RouteDef) -> str:
    signs = {e.sign for e in route.impacts}
    if not signs:
        return ""
    return signs.pop() if len(signs) == 1 else "mixed"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def index_data(ctx: _Context) -> M.IndexData:
    snapshot = ctx.snapshot
    suites = []
    for sf in shown_suite_files(snapshot):
        run_roles = [ctx.roles(r) for r in sf.runs]
        digests = {p.kl_digest for v in sf.versions for p in snapshot.suites[v].passes}
        suites.append(
            M.SuiteListRow(
                name=sf.name,
                key=key_of(sf.name),
                cases=len(sf.cases),
                runs=len(sf.runs),
                suts=sorted({s for r in run_roles for s in r.suts}),
                simulators=sorted({r.simulator for r in run_roles} - {""}),
                checkers=sorted({r.checker for r in run_roles} - {""}),
                judges=sorted({j for r in run_roles for j in r.judges}),
                last_run=local_time(snapshot.runs[sf.runs[0]].created) if sf.runs else "",
                unreadable_case_lines=len(sf.case_problems),
                other_library=bool(
                    snapshot.library and digests and snapshot.library.digest not in digests
                ),
            )
        )
    suites.sort(key=lambda s: (s.last_run, s.name), reverse=True)
    runs = []
    for info in sorted(snapshot.runs.values(), key=lambda r: r.name, reverse=True):
        facts = ctx.run_facts(info.name)
        roles = ctx.roles(info.name)
        runs.append(
            M.RunListRow(
                name=info.name,
                key=key_of(info.name),
                level=info.level,
                suts=roles.suts,
                simulator=roles.simulator,
                checker=roles.checker,
                judges=roles.judges,
                transcripts=len(facts.transcripts),
                buckets=dict(Counter(t.bucket for t in facts.transcripts)),
                created=local_time(info.created),
                suite=info.suite_name if info.suite_name in snapshot.suite_files else None,
            )
        )
    library = snapshot.library
    return M.IndexData(
        built_at=snapshot.built_at.isoformat(timespec="seconds"),
        root=Path(os.path.relpath(snapshot.root, ctx.data_root.parent)).as_posix(),
        library=M.LibraryInfo(version=library.version, digest=library.digest) if library else None,
        library_error=snapshot.library_error,
        suites=suites,
        runs=runs,
    )


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def suite_file_data(ctx: _Context, name: str) -> M.SuiteFileData:
    snapshot = ctx.snapshot
    sf = snapshot.suite_files[name]
    current = {cid: case.record for cid, case in sf.cases.items()}
    versions = []
    for version in sf.versions:
        suite = snapshot.suites[version]
        versions.append(
            M.SuiteVersionData(
                id=version,
                digest=suite.hash,
                current=suite.hash == sf.digest,
                cases=len(suite.cases),
                matches=sum(1 for cid, c in suite.cases.items() if current.get(cid) == c.record),
                runs=[r for r in sf.runs if snapshot.runs[r].suite_id == version],
            )
        )
    runs = []
    transcripts = []
    for run in sf.runs:
        info = snapshot.runs[run]
        facts = ctx.run_facts(run)
        roles = ctx.roles(run)
        created = local_time(info.created)
        runs.append(
            M.SuiteRunRow(
                name=run,
                created=created,
                level=info.level,
                version=info.suite_id or "",
                suts=roles.suts,
                simulator=roles.simulator,
                checker=roles.checker,
                judges=roles.judges,
                transcripts=len(facts.transcripts),
                buckets=dict(Counter(t.bucket for t in facts.transcripts)),
            )
        )
        for t in facts.transcripts:
            if t.sim.error is not None:
                continue
            target = next((r for r in facts.routes(t) if r.is_target), None)
            transcripts.append(
                M.CaseTranscriptRow(
                    test_case_id=t.test_case_id,
                    run=run,
                    id=t.id,
                    created=created,
                    sut=t.sut,
                    simulator=roles.simulator,
                    checker=roles.checker,
                    judge=t.judge.judge_model if t.judge else "",
                    bucket=t.bucket,
                    fidelity="invalid" if t.bucket == "invalid" else t.sim.fidelity,
                    outcome=target.outcome if target else None,
                    same_case=t.case is not None and current.get(t.test_case_id) == t.case.record,
                )
            )
    return M.SuiteFileData(
        name=sf.name,
        key=key_of(sf.name),
        source=ctx.rel(sf.source),
        in_records=sf.path is not None,
        cases=[_case_record(c) for c in sf.cases.values()],
        case_problems=list(sf.case_problems),
        versions=versions,
        runs=runs,
        transcripts=transcripts,
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _target_route(row: F.Route) -> M.TargetData:
    return M.TargetData(
        route=row.route,
        did=row.result.did if row.result else "",
        outcome=row.outcome,
        support=row.result.support if row.result else None,
        gate_status=row.result.gate_status if row.result else None,
    )


def _strip(t: F.Transcript, target: F.Route | None) -> M.StripData:
    result = target.result if target else None
    return M.StripData(
        exchanges=t.sim.exchanges,
        exit_reason=t.sim.exit_reason,
        route_turns=list(result.step_turns) if result else [],
        onset=result.onset if result else None,
        stage1_turns=list(t.judge.stage1_turns) if t.judge else [],
        fidelity_turns=list(t.sim.fidelity_turns),
    )


def run_data(ctx: _Context, name: str) -> M.RunData:
    facts = ctx.run_facts(name)
    info = facts.run
    judges = [t.judge for t in facts.transcripts if t.judge is not None]
    simulation = dict_of(info.manifest.get("simulation"))
    max_turns = simulation.get("max_turns")
    rows = []
    files: dict[str, list[str]] = defaultdict(list)
    for t in facts.transcripts:
        sim = t.sim
        if sim.error is None:
            files[f"{sim.batch}/{sim.file.name}"].append(sim.transcript_id)
        routes = facts.routes(t)
        target = next((r for r in routes if r.is_target), None)
        rows.append(
            M.RunRow(
                id=t.id,
                case_key=key_of(t.run, t.id),
                test_case_id=t.test_case_id,
                display_name=t.case.display_name if t.case else "",
                sut=t.sut,
                batch=sim.batch,
                file=sim.file.name,
                line=sim.line,
                bucket=t.bucket,
                fidelity=sim.fidelity,
                error=sim.error,
                sim_failure=sim.sim_failure,
                fidelity_reasons=list(sim.fidelity_reasons),
                preempted_by=sim.preempted_by,
                level=t.level,
                adherence=list(sim.adherence),
                judged=t.judge is not None,
                target=_target_route(target) if target else None,
                others=[
                    M.RouteRef(route=r.route, did=r.result.did if r.result else "")
                    for r in routes
                    if not r.is_target and r.outcome == "fires"
                ],
                strip=_strip(t, target),
            )
        )
    return M.RunData(
        name=info.name,
        key=key_of(info.name),
        run_id=info.run_id,
        level=info.level,
        suite_id=info.suite_id,
        suite=info.suite_name if info.suite_name in ctx.snapshot.suite_files else None,
        max_turns=max_turns if isinstance(max_turns, int) else None,
        checker_prompt=str(
            dict_of(simulation.get("elicitation_fidelity")).get("prompt_version") or ""
        ),
        passes=[_pass(p) for p in F.passes_of(judges)],
        fidelity_counts=dict(Counter(t.sim.fidelity or "unchecked" for t in facts.transcripts)),
        fact_mode=info.fact_mode,
        manifest=ctx.rel(info.manifest_path),
        cases_file=ctx.rel(info.cases_file),
        batches=[b.name for b in info.batches],
        suts=sorted({t.sut for t in facts.transcripts if t.bucket != "unreadable"}),
        judge_models=sorted({j.judge_model for j in judges if j.judge_model}),
        person_model=str(dict_of(simulation.get("person_response_model")).get("model_id") or ""),
        checker=F.checker_model(info.manifest),
        judge_elapsed_ms=sum(j.elapsed_ms for j in judges),
        judge_tokens=(
            sum(j.tokens[0] for j in judges),
            sum(j.tokens[1] for j in judges),
            sum(j.tokens[2] for j in judges),
        ),
        call_failures=sum(j.call_failures for j in judges),
        retries=sum(j.retries for j in judges),
        judge_problems=_problems(facts.judge_problems),
        orphans=facts.orphans,
        transcripts=rows,
        files=dict(files),
    )


# ---------------------------------------------------------------------------
# Transcripts: quotes, steps and conditions
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = re.sub(r"(\*\*|__|[*_`])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_found(quote: str, content: str) -> bool:
    """Whether ``quote`` appears in ``content``, ignoring markdown markers and spacing."""
    q = _normalize(quote)
    return bool(q) and q in _normalize(content)


Content = Callable[[int, str], str]


def _content_lookup(turns: list[Any]) -> Content:
    contents: dict[tuple[int, str], str] = {}
    for item in turns:
        if isinstance(item, dict):
            number = item.get("turn_number")
            contents[(number if isinstance(number, int) else 0, str(item.get("role") or ""))] = str(
                item.get("content") or ""
            )

    def lookup(turn: int, role: str) -> str:
        if role:
            return contents.get((turn, role), "")
        return " ".join(v for (n, _), v in contents.items() if n == turn)

    return lookup


def _turn(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _cite(span: Mapping[str, Any], content: Content, *, check_verified: bool = True) -> M.Cite:
    """An evidence span or cue as a citation: found when verified and present in its turn."""
    turn = _turn(span.get("turn"))
    role = str(span.get("role") or span.get("speaker") or "")
    quote = str(span.get("quote") or "")
    verified = not check_verified or span.get("verified", True) is not False
    return M.Cite(
        turn=turn,
        quote=quote,
        role=role,
        found=verified and quote_found(quote, content(turn, role)),
    )


_EXCERPT = re.compile(r"'([^']{20,}?)'")


def _normalize_quote(text: str) -> str:
    return " ".join(text.replace("’", "'").split())


def _quoted_in(text: str, quote: str) -> tuple[int, str]:
    """Where ``text`` shows ``quote``: the whole quote, or a '…' excerpt of it. ``-1`` if not."""
    if not quote:
        return -1, ""
    at = text.find(quote)
    if at >= 0:
        return at, quote
    whole = _normalize_quote(quote)
    for match in _EXCERPT.finditer(text):
        piece = match.group(1)
        if _normalize_quote(piece.rstrip(".,;:")) in whole:
            return match.start(1), piece
    return -1, ""


def _evidence_index(stage1: Mapping[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for obs in dicts_of(stage1.get("observations")):
        for span in dicts_of(obs.get("evidence")):
            turn = span.get("turn")
            if isinstance(turn, int):
                index[(str(obs.get("behavior_id") or ""), turn)].append(span)
    return index


def _speaker(behavior: str, did: str, library: F.Library | None) -> str:
    """``user`` or ``assistant``: whose message a behavior is observed in."""
    name = library.name(behavior) if library else None
    kind = name.type if name else ""
    if kind == "user_behavior" or (not kind and did.startswith("UserBehavior")):
        return "user"
    return "assistant"


def _ordered_steps(steps: list[dict[str, Any]], library: F.Library | None) -> list[dict[str, Any]]:
    """Steps in the order they happened: by turn, the person's message before the AI's reply."""

    def key(step: dict[str, Any]) -> tuple[int, int]:
        speaker = _speaker(
            str(step.get("behavior_id") or ""), str(step.get("behavior_did") or ""), library
        )
        return _turn(step.get("turn")), 0 if speaker == "user" else 1

    return sorted(steps, key=key)


def _step(
    step: Mapping[str, Any],
    evidence: Mapping[tuple[str, int], list[dict[str, Any]]],
    content: Content,
) -> M.StepData:
    """One step. A Stage 1 quote the explanation already contains is linked where it stands."""
    behavior = str(step.get("behavior_id") or "")
    turn = step.get("turn") if isinstance(step.get("turn"), int) else -1
    assert isinstance(turn, int)
    contribution = str(step.get("contribution") or "")
    placed: list[tuple[int, int, M.Cite]] = []
    others: list[M.Cite] = []
    for span in evidence.get((behavior, turn), []):
        cite = _cite(span, content)
        start, shown = _quoted_in(contribution, cite.quote)
        overlaps = any(start < end and start + len(shown) > begin for begin, end, _ in placed)
        if start >= 0 and not overlaps:
            placed.append((start, start + len(shown), cite.model_copy(update={"text": shown})))
        else:
            others.append(cite)
    why: list[M.Segment] = []
    cursor = 0
    for begin, end, cite in sorted(placed, key=lambda p: p[0]):
        if contribution[cursor:begin]:
            why.append(M.Segment(text=contribution[cursor:begin]))
        why.append(M.Segment(cite=cite))
        cursor = end
    if contribution[cursor:]:
        why.append(M.Segment(text=contribution[cursor:]))
    return M.StepData(
        turn=turn,
        behavior_id=behavior,
        behavior_did=str(step.get("behavior_did") or ""),
        role=str(step.get("role") or ""),
        why=why,
        others=others,
    )


def _instance(
    number: int,
    inst: Mapping[str, Any],
    library: F.Library | None,
    evidence: Mapping[tuple[str, int], list[dict[str, Any]]],
    content: Content,
) -> M.InstanceData:
    connection = inst.get("connection_evidence")
    rationale = inst.get("rationale")
    note = inst.get("structure_note")
    uncertainty = inst.get("uncertainty")
    return M.InstanceData(
        number=number,
        window=[x for x in inst.get("turn_window") or [] if isinstance(x, int)],
        support=str(inst.get("support") or ""),
        structure=str(inst.get("structure") or ""),
        steps=[
            _step(s, evidence, content)
            for s in _ordered_steps(dicts_of(inst.get("steps")), library)
        ],
        connection_evidence=str(connection) if connection else None,
        rationale=str(rationale) if rationale else None,
        evidence_gaps=[str(g) for g in inst.get("evidence_gaps") or [] if str(g).strip()],
        structure_note=str(note) if note else None,
        uncertainty=float(uncertainty) if isinstance(uncertainty, int | float) else None,
        adjustments=[str(n) for n in inst.get("validation_notes") or [] if str(n).strip()],
    )


# The judge's code writes this when the judge gave no reasoning for the route; it adds
# nothing the occurrences do not show.
_PLACEHOLDER_RATIONALE = re.compile(r"^\d+ instance\(s\) found; strongest is \S+ \(\w+\)\.$")


def _recorded_fact(gate: Mapping[str, Any], facts: list[str], library: F.Library | None) -> str:
    """The recorded context value that decides a context-value gate: same dimension as the gate."""
    if library is None or gate.get("kind") != "context_value":
        return ""
    part = library.name(str(gate.get("part_id") or ""))
    if part is None or not part.dimension:
        return ""
    for value in facts:
        name = library.name(value)
        if name is not None and name.dimension == part.dimension:
            return value
    return ""


def _condition(
    gate: Mapping[str, Any], library: F.Library | None, content: Content, facts: list[str]
) -> M.ConditionData:
    rationale = gate.get("rationale")
    return M.ConditionData(
        part_id=str(gate.get("part_id") or ""),
        part_did=str(gate.get("part_did") or ""),
        label=str(gate.get("label") or ""),
        verdict=str(gate.get("verdict") or "unknown"),
        basis=str(gate.get("basis") or ""),
        recorded=str(gate.get("recorded") or "unknown"),
        inferred=str(gate.get("inferred") or "unknown"),
        fact=_recorded_fact(gate, facts, library),
        conflict=bool(gate.get("conflict")),
        cues=[
            M.CueData(cite=_cite(c, content), reading=str(c.get("reading") or ""))
            for c in dicts_of(gate.get("cues"))
        ],
        rationale=str(rationale) if rationale else None,
    )


def _rail_route(
    a: Mapping[str, Any],
    target: str,
    mismatch: bool,
    library: F.Library | None,
    evidence: Mapping[tuple[str, int], list[dict[str, Any]]],
    content: Content,
    facts: list[str],
) -> M.RailRoute:
    pid = str(a.get("pathway_id") or "")
    instances = dicts_of(a.get("instances"))
    rationale = a.get("rationale")
    return M.RailRoute(
        pathway_id=pid,
        did=str(a.get("pathway_did") or ""),
        label=str(a.get("label") or ""),
        outcome="kl_mismatch" if mismatch else F.route_result(a).outcome,
        is_target=pid == target,
        supported=sum(1 for i in instances if i.get("support") == "supported"),
        partial=sum(1 for i in instances if i.get("support") == "partially_supported"),
        has_condition=bool(dicts_of(a.get("gates"))),
        instances=[
            _instance(n, inst, library, evidence, content)
            for n, inst in enumerate(instances, start=1)
        ],
        missing=[str(m) for m in a.get("missing_participants") or []],
        gates=[_condition(g, library, content, facts) for g in dicts_of(a.get("gates"))],
        rationale=str(rationale)
        if rationale and not _PLACEHOLDER_RATIONALE.match(str(rationale))
        else None,
    )


def _judge_view(
    ctx: _Context,
    t: F.Transcript,
    judge: F.JudgeRow,
    latest: bool,
    sim_record: Mapping[str, Any],
    content: Content,
) -> M.JudgeView:
    library = ctx.library
    rec = F.read_line(judge.file, judge.offset, judge.transcript_id)
    cm = dict_of(rec.get("coralmind_run"))
    stage1, stage2 = dict_of(cm.get("stage1")), dict_of(cm.get("stage2"))
    evidence = _evidence_index(stage1)
    target = t.target
    mismatch = library is not None and judge.kl_digest not in ("", library.digest)
    assessments = sorted(
        dicts_of(stage2.get("assessments")),
        key=lambda a: (a.get("pathway_id") != target, str(a.get("pathway_did"))),
    )
    meta = dict_of(cm.get("run_meta"))
    facts = [str(v) for v in meta.get("facts") or [] if isinstance(v, str)]
    target_missing = None
    if cm and target and target not in {str(a.get("pathway_id")) for a in assessments}:
        target_missing = next(
            (
                str(s.get("reason"))
                for s in dicts_of(stage2.get("skipped"))
                if s.get("pathway_id") == target
            ),
            "absent from the judge output",
        )
    target_parts = []
    if target_missing is not None and library is not None and target in library.routes:
        ruled_out = {
            str(n.get("behavior_id")): str(n.get("reason") or "")
            for n in dicts_of(stage1.get("negatives"))
        }
        target_parts = [
            M.TargetPartData(
                id=p.id,
                seen=p.id in judge.observed,
                failed=p.id in judge.failed_behaviors,
                reason="" if p.id in judge.observed else ruled_out.get(p.id, ""),
            )
            for p in library.routes[target].observations
            if p.kind != "context_value"
        ]
    bucket = F.bucket_of(t.sim, judge, t.level)
    route = None
    if bucket == "judged" and target:
        row = F.route_row(replace(t, judge=judge, bucket=bucket), target, library)
        result = row.result
        route = M.CaseRoute(
            outcome=row.outcome,
            assessment=row.assessment,
            support=result.support if result else None,
            gate_status=result.gate_status if result else None,
            onset=result.onset if result else None,
            step_turns=list(result.step_turns) if result else [],
        )
    calls = _judge_calls_of(judge, rec, sim_record)
    return M.JudgeView(
        ref=_judge_ref(judge),
        judge_model=judge.judge_model,
        pass_key=judge.pass_key if judge.has_meta else None,
        pass_id=F.pass_id(judge.pass_key) if judge.has_meta else None,
        evaluated_at=judge.evaluated_at,
        file=ctx.rel(judge.file) or "",
        line=judge.line,
        trace=ctx.rel(judge_trace_path(judge.file)) or "",
        latest=latest,
        bucket=bucket,
        target_route=route,
        stage1_turns=list(judge.stage1_turns),
        ran=bool(cm),
        status=str(rec.get("status") or ""),
        run_status=str(cm.get("run_status") or ""),
        run_error_detail=str(cm.get("run_error_detail")) if cm.get("run_error_detail") else None,
        routes=[
            _rail_route(a, target, mismatch, library, evidence, content, facts) for a in assessments
        ],
        target_missing=target_missing,
        target_parts=target_parts,
        observations=[
            M.ObservationData(
                behavior_id=str(o.get("behavior_id") or ""),
                behavior_did=str(o.get("behavior_did") or ""),
                label=str(o.get("label") or ""),
                speaker=_speaker(
                    str(o.get("behavior_id") or ""), str(o.get("behavior_did") or ""), library
                ),
                turns=[str(x) for x in o.get("turns") or []],
                evidence=[_cite(s, content) for s in dicts_of(o.get("evidence"))],
            )
            for o in dicts_of(stage1.get("observations"))
        ],
        calls=[
            M.CallRef(
                seq=c.seq,
                stage=c.stage,
                label=c.label,
                prompt_tokens=c.prompt_tokens,
                completion_tokens=c.completion_tokens,
                latency_ms=c.latency_ms,
            )
            for c in calls
        ],
    )


def _judge_ref(judge: F.JudgeRow) -> str:
    return f"{judge.file.name}:{judge.line}"


def _judge_calls_of(
    judge: F.JudgeRow, rec: Mapping[str, Any], sim_record: Mapping[str, Any]
) -> list[JudgeCall]:
    """The calls of judge record ``rec`` on the transcript ``sim_record``, from its trace."""
    meta = dict_of(dict_of(rec.get("coralmind_run")).get("run_meta"))
    users = [
        str(x.get("content") or "")
        for x in dicts_of(sim_record.get("turns"))
        if x.get("role") == "user" and x.get("content")
    ]
    return judge_calls(
        judge_trace_path(judge.file),
        str(sim_record.get("test_case_id") or ""),
        started=str(meta.get("started_at") or ""),
        finished=str(meta.get("finished_at") or ""),
        excerpt=users[-1][:80] if users else "",
    )


# ---------------------------------------------------------------------------
# Transcripts: the case, the turns and the fidelity check
# ---------------------------------------------------------------------------


def _exchanges(turns: list[Any]) -> list[M.ExchangeData]:
    """Messages grouped by exchange, each person message marked when it copies the AI."""
    by_exchange: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in turns:
        if isinstance(item, dict):
            by_exchange[_turn(item.get("turn_number"))].append(item)
    history: list[tuple[str, str]] = []
    numbers: list[int] = []
    out = []
    for number in sorted(by_exchange):
        messages = []
        for item in by_exchange[number]:
            role = str(item.get("role") or "")
            content = item.get("content")
            text = content if isinstance(content, str) else ""
            copied = copied_ai_message(text, history) if role == "user" else None
            history.append((role, text))
            numbers.append(number)
            messages.append(
                M.MessageData(
                    role=role,
                    content=content if isinstance(content, str) else None,
                    copied_from=numbers[copied] if copied is not None else None,
                )
            )
        out.append(M.ExchangeData(number=number, messages=messages))
    return out


def _persona(persona: F.Persona) -> M.PersonaData:
    traits = persona.traits
    history = dict_of(traits.get("clinical_history"))

    def text(value: Any) -> str:
        return "" if value is None else str(value)

    return M.PersonaData(
        description=persona.description,
        clinical_context=persona.clinical_context,
        behavioral_description=persona.behavioral_description,
        age=text(traits.get("age")),
        gender=text(traits.get("gender")),
        presentation=text(history.get("presentation")),
        treatment=text(history.get("current_treatment")),
        insight=text(history.get("insight")),
        engagement=text(history.get("treatment_engagement")),
        elevated=list(persona.elevated),
    )


def _case_record(case: F.Case) -> M.CaseRecordData:
    rec = case.record
    return M.CaseRecordData(
        test_case_id=case.test_case_id,
        display_name=case.display_name,
        description=str(rec.get("description") or ""),
        pathway_id=case.pathway_id,
        persona_ref=case.persona_ref,
        counterpart_of=case.counterpart_of,
        level=case.level,
        situation=list(case.situation),
        design=case.design,
        scenario=str(rec.get("scenario") or ""),
        turns=[t for t in rec.get("turns") or [] if isinstance(t, str)],
        strategies=[
            M.StrategyData(
                strategy=str(s.get("strategy") or ""),
                available_from_turn=str(s.get("available_from_turn") or ""),
                trigger=str(s.get("trigger") or ""),
            )
            for s in dicts_of(rec.get("scenario_strategies"))
        ],
        assertions=[
            M.AssertionData(
                behavior=str(a.get("behavior") or ""), assertion=str(a.get("assertion") or "")
            )
            for a in dicts_of(rec.get("behavior_assertions"))
        ],
        persona=_persona(case.persona) if case.persona else None,
        persona_error=case.persona_error,
        record=dict(rec),
    )


def _fidelity_view(
    record: Mapping[str, Any],
    case: F.Case | None,
    content: Content,
    checked: Mapping[str, Any] | None = None,
    library: F.Library | None = None,
) -> M.FidelityData | None:
    verdict = checked or dict_of(record.get("fidelity"))
    if not verdict:
        return None
    checker = dict_of(verdict.get("checker"))

    def cites(item: Mapping[str, Any]) -> list[M.Cite]:
        out = []
        for c in dicts_of(item.get("citations")):
            turn = _turn(c.get("turn"))
            role = str(c.get("speaker") or "user")
            quote = str(c.get("quote") or "")
            out.append(
                M.Cite(
                    turn=turn, quote=quote, role=role, found=quote_found(quote, content(turn, role))
                )
            )
        return out

    authored = dicts_of((case.record if case else {}).get("scenario_strategies"))

    def strategy_field(number: int, key: str) -> str:
        return str(authored[number - 1].get(key) or "") if 0 < number <= len(authored) else ""

    return M.FidelityData(
        status=str(verdict.get("status") or ""),
        checker_model=str(checker.get("model") or ""),
        checker_prompt_version=str(checker.get("prompt_version") or ""),
        behaviors=[
            M.FidelityBehavior(
                behavior=str(b.get("behavior") or ""),
                catalog_occurred=bool(b.get("catalog_occurred")),
                assertion_occurred=bool(b.get("assertion_occurred")),
                citations=cites(b),
                parts=[
                    M.FidelityPart(
                        part=str(part.get("part") or ""),
                        text=(library.criteria_parts if library else {})
                        .get(str(b.get("behavior") or ""), {})
                        .get(str(part.get("part") or ""), ""),
                        present=bool(part.get("present")),
                        citations=cites(part),
                        ruled_out=next(
                            iter(cites({"citations": [dict_of(part.get("ruled_out_by"))]})), None
                        )
                        if part.get("ruled_out_by")
                        else None,
                        reason=str(part.get("reason") or ""),
                    )
                    for part in dicts_of(b.get("parts"))
                ],
                parts_needed=(library.parts_needed if library else {}).get(
                    str(b.get("behavior") or ""), 0
                ),
                miss=str(b["miss"]) if b.get("miss") else None,
                preempted_by=str(b["preempted_by"]) if b.get("preempted_by") else None,
                ai_citation=cites({"citations": [b["ai_citation"]]})[0]
                if isinstance(b.get("ai_citation"), dict)
                else None,
            )
            for b in dicts_of(verdict.get("behaviors"))
        ],
        facts=[
            M.FidelityFact(
                value=str(f.get("value") or ""),
                mode=str(f.get("mode") or ""),
                presented=bool(f.get("presented")),
                citations=cites(f),
            )
            for f in dicts_of(verdict.get("facts"))
        ],
        strategies=sorted(
            (
                M.FidelityStrategy(
                    number=_turn(s.get("index")),
                    adherence=str(s.get("adherence") or ""),
                    text=strategy_field(_turn(s.get("index")), "strategy"),
                    trigger=strategy_field(_turn(s.get("index")), "trigger"),
                    citations=cites(s),
                    reason=str(s.get("reason") or ""),
                )
                for s in dicts_of(verdict.get("strategies"))
            ),
            key=lambda s: s.number,
        ),
    )


def _check_call(ctx: _Context, sim: F.SimRow, turns: list[Any]) -> M.CheckCallData | None:
    """The fidelity check's prompt and response for this transcript, from its check trace."""
    if sim.error is not None:
        return None
    person = [str(t.get("content") or "") for t in dicts_of(turns) if t.get("role") == "user"]
    row = check_call_from_rows(
        ctx.check_trace_rows(sim.file),
        sim.test_case_id,
        person[-1][:200] if person else "",
    )
    if row is None:
        return None
    return M.CheckCallData(
        model=str(row.get("model") or ""),
        prompt=str(row.get("prompt") or ""),
        response=str(row.get("response") or ""),
        prompt_tokens=_turn(row.get("prompt_tokens")),
        completion_tokens=_turn(row.get("completion_tokens")),
        latency_ms=_turn(row.get("latency_ms")),
    )


def _compare(ctx: _Context, sim: F.SimRow, case: F.Case | None) -> list[M.CompareLink]:
    """The counterfactual counterpart and the same case under other systems, in the suite."""
    snapshot = ctx.snapshot
    info = snapshot.runs[sim.run]
    suite = snapshot.suites.get(info.suite_id) if info.suite_id else None
    if suite is None or case is None:
        return []
    counterparts = {case.counterpart_of} if case.counterpart_of else set()
    counterparts |= {
        c.test_case_id for c in suite.cases.values() if c.counterpart_of == case.test_case_id
    }
    by_case = ctx.suite_sims(suite)
    links = []
    for s in (s for c in sorted(counterparts | {sim.test_case_id}) for s in by_case.get(c, ())):
        if s.test_case_id in counterparts and s.sut_key == sim.sut_key:
            kind = "counterpart"
        elif s.test_case_id == sim.test_case_id and s.sut_key != sim.sut_key:
            kind = "other_sut"
        else:
            continue
        links.append(
            M.CompareLink(
                kind=kind,
                run=s.run,
                transcript=s.transcript_id,
                test_case_id=s.test_case_id,
                sut=snapshot.sut_model(s),
            )
        )
    return links


def case_data(ctx: _Context, sim: F.SimRow) -> M.CaseData:
    snapshot = ctx.snapshot
    t = ctx.transcript(sim)
    record = F.read_line(sim.file, sim.offset, sim.transcript_id) if sim.error is None else {}
    raw_turns = record.get("turns")
    turns = raw_turns if isinstance(raw_turns, list) else None
    content = _content_lookup(turns or [])
    judges = sorted(snapshot.judges_by_key.get(sim.key, ()), key=lambda j: j.evaluated_at)
    latest = judges[-1] if judges else None
    failure = record.get("sim_failure")
    detail = ""
    if isinstance(failure, dict):
        detail = " ".join(
            str(x)
            for x in (
                failure.get("detail"),
                f"at turn {failure['failed_at_turn']}" if failure.get("failed_at_turn") else None,
            )
            if x
        )
    driver = None
    if sim.error is None:
        system, label, prompt = sim_driver_from_rows(
            ctx.sim_trace_rows(sim.file, sim.test_case_id), sim.test_case_id
        )
        if system or prompt:
            driver = M.DriverData(label=label or "Driver prompt", prompt=prompt, system=system)
    check_call = _check_call(ctx, sim, turns or [])
    return M.CaseData(
        run=sim.run,
        transcript_id=sim.transcript_id,
        key=key_of(sim.run, sim.transcript_id),
        test_case_id=sim.test_case_id,
        batch=sim.batch,
        file=ctx.rel(sim.file) or "",
        line=sim.line,
        sut=t.sut,
        sut_key=sim.sut_key,
        simulator=ctx.roles(sim.run).simulator,
        level=t.level,
        fact_mode=t.fact_mode,
        suite_id=t.suite_id,
        suite=snapshot.runs[sim.run].suite_name
        if snapshot.runs[sim.run].suite_name in snapshot.suite_files
        else None,
        error=sim.error,
        sim_failure=sim.sim_failure,
        failure_detail=detail,
        exit_reason=sim.exit_reason,
        exchanges=sim.exchanges,
        fidelity=sim.fidelity,
        # parse_sim puts the copy first among the reasons; the page names it on its own.
        fidelity_reasons=list(
            sim.fidelity_reasons[1:] if sim.copied_turn is not None else sim.fidelity_reasons
        ),
        copied_turn=sim.copied_turn,
        fidelity_turns=list(sim.fidelity_turns),
        case=_case_record(t.case) if t.case else None,
        turns=_exchanges(turns) if turns is not None else None,
        judges=[_judge_view(ctx, t, j, j is latest, record, content) for j in judges],
        unjudged_bucket=F.bucket_of(sim, None, t.level),
        fidelity_view=_fidelity_view(
            record, t.case, content, snapshot.check_verdicts.get(sim.key), snapshot.library
        )
        if sim.error is None
        else None,
        driver=driver,
        check_call=check_call,
        compare=_compare(ctx, sim, t.case),
    )


def records_data(ctx: _Context, sim: F.SimRow) -> M.RecordsData:
    record = F.read_line(sim.file, sim.offset, sim.transcript_id) if sim.error is None else None
    judges: dict[str, M.JudgeRecords] = {}
    for judge in sorted(ctx.snapshot.judges_by_key.get(sim.key, ()), key=lambda j: j.evaluated_at):
        rec = F.read_line(judge.file, judge.offset, judge.transcript_id)
        trace = judge_trace_path(judge.file)
        judges[_judge_ref(judge)] = M.JudgeRecords(
            record={k: v for k, v in rec.items() if k != "judge_prompt_snapshot"},
            prompts=rec.get("judge_prompt_snapshot"),
            calls=[_call_text(trace, c) for c in _judge_calls_of(judge, rec, record or {})],
        )
    return M.RecordsData(transcript_id=sim.transcript_id, transcript=record, judges=judges)


def _call_text(trace: Path, call: JudgeCall) -> M.CallText:
    row = trace_row(trace, call.offset)
    if str(row.get("seq")) != str(call.seq) or row.get("test_case_id") != call.test_case_id:
        raise F.StaleRecord(f"{trace.name} changed since the results were read")

    def text(part: str) -> str:
        value = row.get(part)
        return value if isinstance(value, str) else ""

    return M.CallText(
        seq=call.seq, prompt=text("prompt"), response=text("response"), system=text("system")
    )
