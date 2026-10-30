"""The page data the ingest writes and the Next app reads, one model per generated file.

Fields carry codes, counts and Knowledge Library text. Page wording lives with the TypeScript
components. ``scripts/gen_viewer_types.py`` writes ``viewer/src/lib/generated-types.ts`` from
these models, and ``pixi run check`` fails when the two drift.

Paths are relative to the data repository; no generated file holds an absolute path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_serialization_defaults_required=True,
        use_attribute_docstrings=True,
    )


# ---------------------------------------------------------------------------
# Knowledge Library: library.json
# ---------------------------------------------------------------------------


class EntryNameData(Model):
    did: str
    label: str
    type: str
    definition: str
    dimension: str = ""


class PartData(Model):
    kind: str
    id: str
    role: str


class ImpactData(Model):
    id: str
    sign: str


class RouteDefData(Model):
    """A route's structure from its Knowledge Library entry."""

    id: str
    did: str
    label: str
    definition: str
    parts: list[PartData]
    composition_instruction: str
    neighbors: list[str]
    impacts: list[ImpactData]
    direction: str
    """``harmful`` or ``protective`` when every impact has that sign, ``mixed`` when they
    differ, ``""`` with no impacts."""


class LibraryData(Model):
    """The loaded Knowledge Library: every entry's name, every route's structure, role text."""

    version: str
    digest: str
    names: dict[str, EntryNameData]
    routes: dict[str, RouteDefData]
    roles: dict[str, str]
    """Definition of each composition-part role, keyed by role code."""


# ---------------------------------------------------------------------------
# Index: index.json, status.json, transcripts.json
# ---------------------------------------------------------------------------


class LibraryInfo(Model):
    version: str
    digest: str


class JudgePassData(Model):
    """One judge setup: records sharing a judge model, prompt and Knowledge Library digest."""

    key: str
    id: str
    judge_model: str
    prompt_version: str
    prompt_hash: str
    kl_digest: str
    kl_version: str
    transcripts: int
    latest: str


class SuiteListRow(Model):
    """One test case file on the Suites list."""

    name: str
    """The file's path under ``test_case_records/``, without ``.jsonl``."""
    key: str
    """The file key of ``suite-files/<key>.json``."""
    cases: int
    """Test cases in the current file."""
    runs: int
    suts: list[str]
    simulators: list[str]
    checkers: list[str]
    judges: list[str]
    last_run: str
    """The newest run's creation time, ISO local time without a zone; ``""`` with no runs."""
    unreadable_case_lines: int
    other_library: bool
    """Judged with a Knowledge Library other than the one loaded."""


class RunListRow(Model):
    name: str
    key: str
    level: int | None
    suts: list[str]
    simulator: str
    checker: str
    judges: list[str]
    transcripts: int
    buckets: dict[str, int]
    created: str
    """ISO local time without a zone; ``""`` if unknown."""
    suite: str | None
    """The run's suite name; None when its test cases were not copied into the run."""


class IndexData(Model):
    built_at: str
    """ISO time, in UTC, that the ingest read the results."""
    root: str
    """The results folder, relative to the folder that holds the data repository."""
    library: LibraryInfo | None
    library_error: str | None
    suites: list[SuiteListRow]
    runs: list[RunListRow]


class StatusData(Model):
    """Written by every ingest; a failed re-ingest records why here and keeps the last good data."""

    stale_since: str | None
    error: str | None


class TranscriptPlace(Model):
    run: str
    key: str
    """The file key of ``cases/<key>.json``."""


class TranscriptsData(Model):
    """Where each transcript id lives: one entry per run that holds it, oldest run first."""

    places: dict[str, list[TranscriptPlace]]


# ---------------------------------------------------------------------------
# Runs: runs/<key>.json
# ---------------------------------------------------------------------------


class FileProblem(Model):
    file: str
    """File name only."""
    line: int
    error: str


class StripData(Model):
    exchanges: int
    exit_reason: str | None
    route_turns: list[int]
    onset: int | None
    stage1_turns: list[int]
    fidelity_turns: list[int]


class TargetData(Model):
    route: str
    did: str
    outcome: str
    support: str | None
    gate_status: str | None


class RouteRef(Model):
    route: str
    did: str


class RunRow(Model):
    id: str
    case_key: str
    test_case_id: str
    display_name: str
    """The test case's name; ``""`` when it has none or is not in the suite."""
    sut: str
    batch: str
    file: str
    """File name only."""
    line: int
    bucket: str
    fidelity: str | None
    error: str | None
    sim_failure: str | None
    fidelity_reasons: list[tuple[str, str]]
    preempted_by: str | None
    level: int | None
    adherence: list[str]
    judged: bool
    """Whether a judge record exists for the transcript."""
    target: TargetData | None
    others: list[RouteRef]
    """Other routes that fired."""
    strip: StripData


class RunData(Model):
    name: str
    key: str
    run_id: str
    level: int | None
    suite_id: str | None
    suite: str | None
    max_turns: int | None
    checker_prompt: str
    passes: list[JudgePassData]
    """The run's judge passes, the one with the most transcripts first."""
    fidelity_counts: dict[str, int]
    """Transcripts by fidelity status; ``unchecked`` for none."""
    fact_mode: str
    manifest: str | None
    cases_file: str | None
    batches: list[str]
    suts: list[str]
    judge_models: list[str]
    person_model: str
    checker: str
    judge_elapsed_ms: int
    judge_tokens: tuple[int, int, int]
    call_failures: int
    retries: int
    judge_problems: list[FileProblem]
    orphans: int
    transcripts: list[RunRow]
    files: dict[str, list[str]]
    """``<batch>/<file name>`` → the transcript ids of its readable lines, in file order."""


# ---------------------------------------------------------------------------
# Transcripts: cases/<key>.json
# ---------------------------------------------------------------------------


class Cite(Model):
    """A citation of a turn: the quote, whose message, and whether it is in that message."""

    turn: int
    quote: str
    role: str
    found: bool
    text: str | None = None
    """Words of an explanation that quote this span, linked in place of the quote."""


class Segment(Model):
    """A piece of a step's explanation: plain text, or a citation linked where it stands."""

    text: str | None = None
    cite: Cite | None = None


class StepData(Model):
    turn: int
    behavior_id: str
    behavior_did: str
    role: str
    why: list[Segment]
    others: list[Cite]


class InstanceData(Model):
    number: int
    window: list[int]
    support: str
    structure: str
    steps: list[StepData]
    connection_evidence: str | None
    rationale: str | None
    """The judge's reasoning on this occurrence."""
    evidence_gaps: list[str]
    """What the judge found missing for the occurrence to be supported."""
    structure_note: str | None
    uncertainty: float | None
    """The judge's own uncertainty, from 0 (certain) to 1."""
    adjustments: list[str]
    """Changes the judge's code made to the judge's answer, such as a support level it replaced."""


class CueData(Model):
    cite: Cite
    reading: str


class ConditionData(Model):
    part_id: str
    part_did: str
    label: str
    verdict: str
    basis: str
    recorded: str
    inferred: str
    fact: str
    """The person's recorded value on the condition's dimension, or ``""``."""
    conflict: bool
    cues: list[CueData]
    rationale: str | None


class RailRoute(Model):
    pathway_id: str
    did: str
    label: str
    outcome: str
    is_target: bool
    supported: int
    partial: int
    has_condition: bool
    instances: list[InstanceData]
    missing: list[str]
    gates: list[ConditionData]
    rationale: str | None


class ObservationData(Model):
    behavior_id: str
    behavior_did: str
    label: str
    speaker: str
    """``user`` or ``assistant``: whose messages the behavior is observed in."""
    turns: list[str]
    evidence: list[Cite]


class CallRef(Model):
    seq: int
    stage: int
    label: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class CaseRoute(Model):
    """The target route's row for the verdict line and the turn strip."""

    outcome: str
    assessment: str
    support: str | None
    gate_status: str | None
    onset: int | None
    step_turns: list[int]


class TargetPartData(Model):
    """One behavior of the target pathway, and whether the judge's Stage 1 saw it."""

    id: str
    seen: bool
    failed: bool
    """The Stage 1 call that would have read it failed."""
    reason: str = ""
    """The judge's reason for ruling it out in Stage 1; empty when seen or not recorded."""


class JudgeView(Model):
    """One judge record of the transcript, read for the rail."""

    ref: str
    """``<file name>:<line>``, naming the record in links."""
    judge_model: str
    pass_key: str | None
    pass_id: str | None
    evaluated_at: str
    file: str
    line: int
    trace: str
    latest: bool
    bucket: str
    target_route: CaseRoute | None
    stage1_turns: list[int]
    ran: bool
    status: str
    run_status: str
    run_error_detail: str | None
    routes: list[RailRoute]
    target_missing: str | None
    """Why the target has no assessment, when it has none."""
    target_parts: list[TargetPartData]
    """The target's behaviors and whether Stage 1 saw each; set only with ``target_missing``."""
    observations: list[ObservationData]
    calls: list[CallRef]


class FidelityPart(Model):
    part: str
    """The part's letter in the behavior's criteria, or ``all``."""
    text: str
    """The lettered clause from the Knowledge Library; ``""`` for ``all`` or when unknown."""
    present: bool
    citations: list[Cite]
    ruled_out: Cite | None
    """The person turn that breaks a part not present; None when no turn does."""
    reason: str
    """The checker's one-sentence account of the part, unverified; ``""`` without one."""


class FidelityBehavior(Model):
    behavior: str
    catalog_occurred: bool
    assertion_occurred: bool
    citations: list[Cite]
    parts: list[FidelityPart]
    parts_needed: int
    """How many of its lettered parts the behavior needs, 1 when any one counts; 0 when unknown."""
    miss: str | None
    """``not_attempted`` or ``preempted`` when the behavior did not occur."""
    preempted_by: str | None
    """``resolved`` or ``derailed`` for a preempted miss."""
    ai_citation: Cite | None
    """The AI turn that made the behavior unnecessary."""


class FidelityFact(Model):
    value: str
    mode: str
    presented: bool
    citations: list[Cite]


class FidelityStrategy(Model):
    number: int
    """The strategy's 1-based position in the test case's ``scenario_strategies``."""
    adherence: str
    text: str
    trigger: str
    citations: list[Cite]
    """For ``followed``, the person's turn; for ``not_followed``, the trigger's turn."""
    reason: str
    """The checker's one-sentence account of the adherence, unverified; ``""`` without one."""


class FidelityData(Model):
    status: str
    checker_model: str
    checker_prompt_version: str
    behaviors: list[FidelityBehavior]
    facts: list[FidelityFact]
    strategies: list[FidelityStrategy]


class MessageData(Model):
    role: str
    content: str | None
    copied_from: int | None
    """The turn of the AI message this person message repeats."""


class ExchangeData(Model):
    number: int
    messages: list[MessageData]


class PersonaData(Model):
    description: str
    clinical_context: str
    behavioral_description: str
    age: str
    gender: str
    presentation: str
    treatment: str
    insight: str
    engagement: str
    elevated: list[str]


class StrategyData(Model):
    strategy: str
    available_from_turn: str
    trigger: str


class AssertionData(Model):
    behavior: str
    assertion: str


class CaseRecordData(Model):
    test_case_id: str
    display_name: str
    description: str
    pathway_id: str
    persona_ref: str
    counterpart_of: str | None
    level: int | None
    situation: list[tuple[str, str]]
    design: tuple[str, int] | None
    scenario: str
    turns: list[str]
    strategies: list[StrategyData]
    assertions: list[AssertionData]
    persona: PersonaData | None
    persona_error: str | None
    record: dict[str, Any]


class SuiteVersionData(Model):
    """One version of a test case file that runs used."""

    id: str
    """The suite id ``<name>@<digest>``."""
    digest: str
    current: bool
    """Whether this version is the current file."""
    cases: int
    matches: int
    """Test cases in this version that are unchanged in the current file."""
    runs: list[str]


class SuiteRunRow(Model):
    name: str
    created: str
    level: int | None
    version: str
    """The suite id of the version the run used."""
    suts: list[str]
    simulator: str
    checker: str
    judges: list[str]
    transcripts: int
    buckets: dict[str, int]


class CaseTranscriptRow(Model):
    """A transcript made from one of the suite's test cases, in any run."""

    test_case_id: str
    run: str
    id: str
    created: str
    sut: str
    simulator: str
    checker: str
    judge: str
    """The judge model of the transcript's latest judge record; ``""`` when not judged."""
    bucket: str
    fidelity: str | None
    """The fidelity status; ``invalid`` also when the person copied an AI message."""
    outcome: str | None
    """The target route's outcome when judged."""
    same_case: bool
    """Whether the run used the test case as the current file has it."""


class SuiteFileData(Model):
    """A test case file, its versions, and every run and transcript made from it."""

    name: str
    key: str
    source: str | None
    """Where the current test cases were read, relative to the data repository."""
    in_records: bool
    """Whether ``source`` is the file in ``test_case_records/`` rather than a run's copy."""
    cases: list[CaseRecordData]
    case_problems: list[str]
    versions: list[SuiteVersionData]
    runs: list[SuiteRunRow]
    transcripts: list[CaseTranscriptRow]


class CompareLink(Model):
    kind: Literal["counterpart", "other_sut"]
    run: str
    transcript: str
    test_case_id: str
    sut: str


class DriverData(Model):
    label: str
    prompt: str
    system: str | None


class CheckCallData(Model):
    """The fidelity check's model call on one transcript, from its check trace."""

    model: str
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class CaseData(Model):
    run: str
    transcript_id: str
    key: str
    test_case_id: str
    batch: str
    file: str
    line: int
    sut: str
    sut_key: str
    simulator: str
    """The person simulator's model; ``""`` for a scripted run."""
    level: int | None
    fact_mode: str
    suite_id: str | None
    suite: str | None
    error: str | None
    sim_failure: str | None
    failure_detail: str
    exit_reason: str | None
    exchanges: int
    fidelity: str | None
    fidelity_reasons: list[tuple[str, str]]
    """The fidelity check's own reasons."""
    copied_turn: int | None
    """The first person turn that copies an AI message; it makes the transcript invalid."""
    fidelity_turns: list[int]
    case: CaseRecordData | None
    turns: list[ExchangeData] | None
    """None when the record holds no turns list."""
    judges: list[JudgeView]
    unjudged_bucket: str
    """The bucket when no judge record is used."""
    fidelity_view: FidelityData | None
    driver: DriverData | None
    check_call: CheckCallData | None
    compare: list[CompareLink]


# ---------------------------------------------------------------------------
# Records: records/<key>.json, what /record and /judge-call serve
# ---------------------------------------------------------------------------


class CallText(Model):
    seq: int
    prompt: str
    response: str
    system: str


class JudgeRecords(Model):
    record: dict[str, Any]
    """The judge record without its prompt snapshot."""
    prompts: Any
    """The record's ``judge_prompt_snapshot``; None when it has none."""
    calls: list[CallText]
    """The calls :class:`JudgeView` lists, with their text from the judge trace."""


class RecordsData(Model):
    """The records of one transcript, copied as the ingest read them, keyed like its case."""

    transcript_id: str
    transcript: dict[str, Any] | None
    """The transcript record; None when its line could not be read."""
    judges: dict[str, JudgeRecords]
    """By :attr:`JudgeView.ref`."""


GENERATED_MODELS: tuple[type[Model], ...] = (
    LibraryData,
    IndexData,
    StatusData,
    TranscriptsData,
    SuiteFileData,
    RunData,
    CaseData,
    RecordsData,
)
"""The root model of every generated file, in the order types are written."""
