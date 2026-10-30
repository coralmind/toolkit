"""Result schemas for the CoralMind two-stage judge.

Stage 1 (behavior identification) and Stage 2 (pathway assessment) are stored
separately so a wrong conclusion can be traced to behavior identification,
candidate selection, or pathway assessment. Three kinds of statement are kept
apart by construction:

* **Behavioral observations** (:class:`BehaviorObservation`) — things that can be
  located, quoted, and verified in the transcript. ``evidence[].verified`` records
  whether the quoted span was actually found in the cited turn.
* **Pathway assessments** (:class:`PathwayAssessment`) — interpretive hypotheses
  relative to one knowledge-base definition, expressed as turn-anchored
  :class:`PathwayInstance` records (the same pathway at turns 1–2 and again at
  turns 50–51 is two instances). ``support`` says whether the observations at
  those turns satisfy the definition; it never asserts that harm occurred.
* **Gate assessments** (:class:`GateAssessment`) -- whether each fact a pathway
  requires held in this conversation, read from what the test case records and
  from verbatim cues in the transcript, kept apart so a disagreement between the
  two stays visible. A pathway applies only where its configuration is
  instantiated and every gate holds.

Impacts are out of scope: the judge reports behaviors and whether a pathway
configuration is instantiated, never whether an impact occurred.

Everything the run needs for traceability (hashes, model, prompt version,
coverage, call/failure/retry counts, tokens, elapsed time) is on
:class:`CoralMindRunMeta`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from corallab.schemas.core.kl import KlPin

__all__ = [
    "BehaviorNegative",
    "BehaviorObservation",
    "CoralMindRun",
    "CoralMindRunMeta",
    "EvidenceSpan",
    "GateAssessment",
    "GateCue",
    "InstanceStep",
    "KlPin",
    "PathwayAssessment",
    "PathwayInstance",
    "PathwaySkip",
    "Stage1CallRecord",
    "Stage1Result",
    "Stage2CallRecord",
    "Stage2Result",
    "SupportingBehavior",
    "TurnBehaviorEntry",
    "TurnBehaviorSummary",
]

SupportLevel = Literal["supported", "partially_supported", "not_supported", "insufficient_evidence"]
GateReading = Literal["holds", "does_not_hold", "unknown"]
GateStatus = Literal["no_gates", "met", "not_met", "unknown"]
CallStatus = Literal["success", "judge_error", "dry_run"]


class EvidenceSpan(BaseModel):
    """A quoted span located in one transcript turn."""

    turn: int = Field(description="1-indexed turn number the quote was taken from")
    role: str | None = Field(default=None, description="Speaker role of that turn")
    quote: str = Field(description="Verbatim excerpt as returned by the judge")
    verified: bool = Field(
        default=False,
        description="True when the quote was found (whitespace-normalized) in the cited turn",
    )


class BehaviorObservation(BaseModel):
    """One behavior definition found in the transcript (Stage 1 output)."""

    behavior_id: str = Field(description="Knowledge Library id, e.g. 'kl-900001'")
    behavior_did: str = Field(description="Display id, e.g. 'AiBehavior-1' / 'UserBehavior-3'")
    aliases: list[str] = Field(default_factory=list, description="Aliases from the catalog entry")
    label: str = Field(description="Behavior title from the knowledge base")
    kind: Literal["ai_behavior", "user_behavior"] = Field(description="Whose behavior it is")
    detection_window: str | None = Field(
        default=None, description="single_turn / multi_turn / cross_session per the definition"
    )
    turns: list[int] = Field(default_factory=list, description="Turns where the behavior occurs")
    evidence: list[EvidenceSpan] = Field(default_factory=list, description="Quoted spans")
    uncertainty: float = Field(
        ge=0.0, le=1.0, description="Judge's residual doubt (0 = criteria unambiguously met)"
    )
    rationale: str = Field(default="", description="Why the definition's criteria are met")
    group_id: str = Field(description="Stage-1 definition group that produced this observation")
    validation_notes: list[str] = Field(
        default_factory=list,
        description="Post-hoc checks that did not pass (unverified quote, wrong speaker, …)",
    )


class BehaviorNegative(BaseModel):
    """A definition the judge explicitly marked absent (Stage 1 is a checklist:
    every definition gets a verdict, so a miss is a recorded decision, not silence)."""

    behavior_id: str = Field(description="Behavior node id")
    behavior_did: str = Field(description="Display id")
    label: str = Field(description="Behavior title")
    kind: Literal["ai_behavior", "user_behavior"]
    reason: str = Field(default="", description="Judge's one-sentence reason for absence")
    group_id: str = Field(description="Stage-1 definition group that produced this verdict")


class TurnBehaviorEntry(BaseModel):
    """One behavior anchored to one turn, with the evidence quoted from that turn."""

    behavior_id: str = Field(description="Behavior node id")
    behavior_did: str = Field(description="Display id, e.g. 'AiBehavior-18'")
    label: str = Field(description="Behavior title")
    kind: Literal["ai_behavior", "user_behavior"] = Field(
        description="Matches the turn's speaker: AI behaviors on assistant turns, user on user"
    )
    quotes: list[str] = Field(
        default_factory=list, description="Verified quotes from this turn supporting the behavior"
    )
    uncertainty: float = Field(ge=0.0, le=1.0, description="Stage-1 uncertainty")
    rationale: str = Field(default="", description="Stage-1 rationale for the behavior")


class TurnBehaviorSummary(BaseModel):
    """Turn-anchored view of Stage 1: what each turn's speaker did, in transcript order.

    The primary reading of Stage 1. A USER turn lists user behaviors only; an
    AI ASSISTANT turn lists AI behaviors only — evidence anchored to the wrong
    speaker is discarded during normalization, not recorded here.
    """

    turn: int = Field(description="1-indexed turn number")
    role: str = Field(description="Speaker role of the turn")
    behavior_ids: list[str] = Field(
        default_factory=list, description="Behavior node ids observed at this turn"
    )
    behaviors: list[TurnBehaviorEntry] = Field(
        default_factory=list, description="Behaviors observed at this turn, with their quotes"
    )


class Stage1CallRecord(BaseModel):
    """Accounting for one Stage-1 LLM call (one definition group, full transcript)."""

    group_id: str = Field(description="Definition group id, e.g. 'ai_behavior-g03'")
    kind: str = Field(description="'ai_behavior' or 'user_behavior'")
    group_labels: list[str] = Field(
        default_factory=list, description="Website group labels merged into this call"
    )
    behavior_ids: list[str] = Field(default_factory=list, description="Definitions evaluated")
    related_behavior_ids: list[str] = Field(
        default_factory=list,
        description="Reference-only definitions included for disambiguation (not reportable)",
    )
    unassessed_ids: list[str] = Field(
        default_factory=list,
        description="Definitions of this group the judge returned no verdict for",
    )
    status: CallStatus = Field(description="success / judge_error / dry_run")
    attempts: int = Field(default=0, description="LLM calls made including retries")
    retries: int = Field(default=0, description="attempts - 1 when attempts > 0")
    validation_failures: int = Field(
        default=0, description="Responses that parsed but failed output validation"
    )
    error: str | None = Field(default=None, description="Terminal error, if any")
    prompt_tokens: int | None = Field(default=None, description="Sum over attempts")
    completion_tokens: int | None = Field(default=None, description="Sum over attempts")
    cached_tokens: int | None = Field(default=None, description="Sum over attempts")
    latency_ms: int | None = Field(default=None, description="Wall time of the call")
    prompt_chars: int = Field(default=0, description="Length of the assembled prompt")
    raw_response: str | None = Field(
        default=None, description="Last raw response text, kept only when the call failed"
    )


class Stage1Result(BaseModel):
    """Behavior identification over the whole transcript."""

    observations: list[BehaviorObservation] = Field(default_factory=list)
    per_turn: list[TurnBehaviorSummary] = Field(
        default_factory=list,
        description=(
            "Turn-anchored timeline (every turn listed, in order): the speaker's own "
            "behaviors observed at that turn. This is the primary Stage-1 view; "
            "`observations` is the same data grouped by behavior."
        ),
    )
    negatives: list[BehaviorNegative] = Field(
        default_factory=list,
        description="Definitions explicitly judged absent, with the judge's reason",
    )
    unassessed: list[str] = Field(
        default_factory=list,
        description="Definitions for which a successful call returned no verdict at all",
    )
    calls: list[Stage1CallRecord] = Field(default_factory=list)
    coverage: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per kind: {'evaluated': n, 'total': n, 'failed_groups': n}",
    )
    dropped: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Judge items discarded during normalization, with the reason",
    )
    integrity: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Evidence integrity counts: definitions_in_calls, definitions_present, "
            "definitions_absent, definitions_unassessed, turns_claimed, turns_kept, "
            "turns_dropped_wrong_speaker, turns_dropped_unverified, quotes_total, "
            "quotes_verified. Every kept turn has a verified quote (unless the behavior "
            "is absence-type)."
        ),
    )


class SupportingBehavior(BaseModel):
    """A Stage-1 observation the judge cites for a pathway."""

    behavior_id: str = Field(description="Behavior node id")
    behavior_did: str | None = Field(default=None, description="Display id")
    role: str | None = Field(
        default=None, description="Participant role per the pathway definition, if any"
    )
    turns: list[int] = Field(default_factory=list, description="Turns cited")
    is_participant: bool = Field(
        default=True,
        description="False when the judge cited a behavior that is not a defined participant",
    )


InstanceStructure = Literal["single_turn", "pair", "sequence", "accumulation", "conjunction"]


class InstanceStep(BaseModel):
    """One element of a pathway instance: one behavior at one turn, in order."""

    order: int = Field(description="1-based position in the instance")
    turn: int = Field(description="Turn number (anchored to the behavior's speaker)")
    behavior_id: str = Field(description="Behavior node id")
    behavior_did: str | None = Field(default=None, description="Display id")
    role: str | None = Field(default=None, description="Participant role at this step")
    contribution: str = Field(
        default="", description="What this turn contributes to the occurrence"
    )


class PathwayInstance(BaseModel):
    """One concrete occurrence of a pathway, anchored to a specific turn window.

    The same pathway can be instantiated several times in one transcript (turns
    1–2 and again turns 50–51); each occurrence is its own instance and is never
    merged with the others.
    """

    instance_id: str = Field(description="'<pathway_did>#<n>', numbered by turn window")
    turn_window: list[int] = Field(
        description="[first turn, last turn] of the turns this instance uses"
    )
    turns: list[int] = Field(
        default_factory=list, description="Every turn any participant of this instance uses"
    )
    support: SupportLevel = Field(
        description=(
            "Does this occurrence satisfy the pathway's definition? About the definition "
            "being instantiated at these turns, never about an outcome."
        )
    )
    structure: InstanceStructure = Field(
        default="sequence",
        description=(
            "How the steps form one occurrence: single_turn (one turn is the whole "
            "occurrence), pair (initiating turn + linked response), sequence (ordered "
            "participants across turns), accumulation (a multi_turn / trajectory "
            "definition that only exists as a set of turns), conjunction (co-occurring "
            "elements within a window)"
        ),
    )
    structure_note: str = Field(
        default="",
        description="Why these steps are one occurrence and which definition requires it",
    )
    steps: list[InstanceStep] = Field(
        default_factory=list, description="Ordered (behavior, turn) elements of the occurrence"
    )
    participants: list[SupportingBehavior] = Field(
        default_factory=list,
        description="Behaviors forming this instance, each with only the turns of this instance",
    )
    connection_evidence: str = Field(
        default="",
        description="How roles, ordering and temporal/combination requirements are met here",
    )
    evidence_gaps: list[str] = Field(default_factory=list)
    uncertainty: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")
    validation_notes: list[str] = Field(default_factory=list)


class GateCue(BaseModel):
    """A verbatim span of the transcript that bears on whether a gate holds."""

    turn: int = Field(description="1-indexed turn number the quote was taken from")
    speaker: Literal["user", "assistant"] = Field(description="Who spoke the turn")
    quote: str = Field(description="Verbatim excerpt, found in the cited turn")
    reading: Literal["supports", "contradicts"] = Field(
        description="Whether the span is evidence that the gate, as stated, holds or does not"
    )


class GateAssessment(BaseModel):
    """Whether one of a pathway's gates held in this conversation, and on what evidence.

    ``recorded`` is what the test case's facts say, decided by
    ``coralmap.models.holds_for_test_case`` with no model call. ``inferred`` is the
    judge's reading of the conversation alone, backed by ``cues``. ``verdict`` takes
    ``recorded`` when it is not ``unknown`` and ``inferred`` otherwise, and ``conflict``
    marks a transcript that reads the other way from the record.
    """

    gate: str = Field(description="Position of the gate in the pathway's prompt, 'G1', 'G2', ...")
    part_id: str = Field(
        description="The gate's context value or behavior id, or 'OWED' for an owed behavior"
    )
    part_did: str | None = Field(default=None, description="Display id of the gate's entry")
    label: str = Field(description="The gate's label, or the owed behavior's text")
    kind: Literal["context_value", "ai_behavior", "user_behavior"]
    negated: bool = Field(
        default=False,
        description=(
            "True for an `absent` part: the gate holds when the test case records a value on "
            "the dimension and it is not this one"
        ),
    )
    recorded: GateReading = Field(
        description=(
            "What the test case's facts say about the gate as stated. 'unknown' when nothing is "
            "recorded on the value's dimension, and always for a behavior gate"
        )
    )
    cues: list[GateCue] = Field(
        default_factory=list, description="Verified transcript spans the judge relied on"
    )
    inferred: GateReading = Field(
        description=(
            "The judge's reading of the conversation alone. 'unknown' when no verified cue "
            "decides it"
        )
    )
    verdict: GateReading = Field(description="`recorded` when known, otherwise `inferred`")
    basis: Literal["test_case", "conversation", "none"] = Field(
        description="Which of `recorded` and `inferred` decided `verdict`"
    )
    conflict: bool = Field(
        default=False,
        description=(
            "True when `recorded` and `inferred` are both known and disagree: the conversation "
            "reads against the test case, which a fidelity review should look at"
        ),
    )
    rationale: str = Field(default="", description="The judge's reasoning about this gate")
    validation_notes: list[str] = Field(default_factory=list)


class PathwayAssessment(BaseModel):
    """Stage-2 verdict on one pathway hypothesis: its turn-anchored instances.

    ``instances`` is the primary content. When it is empty, ``support`` /
    ``missing_participants`` / ``evidence_gaps`` / ``rationale`` explain why no
    instance could be formed; when it is not, ``support`` is the strongest
    instance's verdict and the pathway-level fields summarize the instances.
    """

    pathway_id: str = Field(description="Pathway node id, e.g. 'kl-000053'")
    pathway_did: str = Field(description="Display id, e.g. 'Pathway-15'")
    aliases: list[str] = Field(default_factory=list, description="Aliases from the catalog entry")
    label: str = Field(description="Pathway title")
    instances: list[PathwayInstance] = Field(
        default_factory=list, description="Turn-anchored occurrences, ordered by turn window"
    )
    support: SupportLevel = Field(
        description=(
            "Strongest instance verdict, or the no-instance verdict "
            "(not_supported / insufficient_evidence) when no instance could be formed"
        )
    )
    missing_participants: list[str] = Field(
        default_factory=list,
        description="Participant behavior ids the definition needs that were never observed",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list, description="Pathway-level gaps (why no instance, if none)"
    )
    uncertainty: float = Field(ge=0.0, le=1.0, description="Residual doubt in `support`")
    rationale: str = Field(default="", description="Pathway-level reasoning")
    candidate_reason: str = Field(
        default="", description="Why this pathway was selected for assessment"
    )
    gates: list[GateAssessment] = Field(
        default_factory=list, description="One per gate of the pathway, in authored order"
    )
    gate_status: GateStatus = Field(
        default="no_gates",
        description=(
            "'not_met' when any gate's verdict is does_not_hold, otherwise 'unknown' when any is "
            "unknown, otherwise 'met'; 'no_gates' for a pathway without gates. `support` says "
            "whether the configuration occurred and this says whether the pathway applies"
        ),
    )
    gate_rationale: str = Field(
        default="", description="The judge's reasoning about the gates taken together"
    )
    validation_notes: list[str] = Field(default_factory=list)


class PathwaySkip(BaseModel):
    """A pathway that was not sent to Stage 2."""

    pathway_id: str
    pathway_did: str
    reason: str


class Stage2CallRecord(BaseModel):
    """Accounting for one Stage-2 LLM call."""

    pathway_ids: list[str] = Field(default_factory=list)
    status: CallStatus
    attempts: int = 0
    retries: int = 0
    validation_failures: int = 0
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    latency_ms: int | None = None
    prompt_chars: int = 0
    raw_response: str | None = None


class Stage2Result(BaseModel):
    """Pathway assessment over the candidate set."""

    assessments: list[PathwayAssessment] = Field(default_factory=list)
    skipped: list[PathwaySkip] = Field(default_factory=list)
    calls: list[Stage2CallRecord] = Field(default_factory=list)
    coverage: dict[str, int] = Field(
        default_factory=dict,
        description="{'total': n, 'candidates': n, 'assessed': n, 'skipped': n, 'failed': n}",
    )
    integrity: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Verdict-completeness counters: candidates, assessed, pathways_missing, "
            "pathways_missing_in_batch, pathways_recovered_by_followup, "
            "pathways_failed_calls, instances_total, no_instance_verdicts, "
            "and support_<level> counts"
        ),
    )
    dropped: list[dict[str, Any]] = Field(default_factory=list)


class CoralMindRunMeta(BaseModel):
    """Run metadata and traceability."""

    run_id: str = Field(description="Unique id of this judge run (one transcript)")
    started_at: str = Field(description="ISO 8601 UTC")
    finished_at: str = Field(description="ISO 8601 UTC")
    elapsed_ms: int = Field(description="Wall time for both stages")
    mode: Literal["live", "dry_run", "mock"] = Field(
        description=(
            "live = real model calls; dry_run = prompts assembled, no calls; mock = test double"
        )
    )
    test_case_id: str = Field(description="Transcript's test case id")
    transcript_hash: str = Field(description="SHA-256 of the canonical transcript JSON")
    transcript_turns: int = Field(description="Number of turns")
    kl: KlPin = Field(
        description=("Which Knowledge Library produced this run (package version and entry digest)")
    )
    model: str = Field(description="Judge model id")
    prompt_version: str = Field(description="Prompt template version directory, e.g. 'v1'")
    prompt_hash: str = Field(description="SHA-256 over the prompt template files")
    config: dict[str, Any] = Field(default_factory=dict, description="Effective judge options")
    facts: list[str] = Field(
        default_factory=list,
        description="Context value ids the test case records, which the gates were checked against",
    )
    coverage: dict[str, Any] = Field(
        default_factory=dict, description="Behavior + pathway evaluation coverage"
    )
    counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "stage1_calls, stage2_calls, candidates, skipped, call_failures, "
            "validation_failures, retries, behaviors_observed, "
            "behaviors_judged_absent, behaviors_unassessed, pathways_assessed, "
            "pathways_missing, pathway_instances"
        ),
    )
    tokens: dict[str, int] | None = Field(
        default=None, description="{'prompt', 'completion', 'cached'} summed over all calls"
    )
    cost_usd: float | None = Field(
        default=None,
        description="Only computed when a pricing source was supplied in the judge options",
    )
    pricing_source: str | None = Field(default=None, description="Where the prices came from")
    cost_note: str = Field(default="", description="Why cost is or is not populated")


class CoralMindRun(BaseModel):
    """Complete output of the two-stage judge for one transcript."""

    run_status: Literal["success", "judge_error", "dry_run"] = Field(default="success")
    run_error_detail: str | None = Field(default=None)
    stage1: Stage1Result = Field(default_factory=Stage1Result)
    stage2: Stage2Result = Field(default_factory=Stage2Result)
    run_meta: CoralMindRunMeta
