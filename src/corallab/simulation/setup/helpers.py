"""Shared helpers for the simulation runners."""

from __future__ import annotations

from corallab.code_version import produced_by
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import (
    ConversationEndReason,
    SimFailure,
    Transcript,
)


def make_transcript(
    case: RunCase,
    manifest: RunManifest,
    *,
    turns: list[Turn] | None = None,
    sim_failure: SimFailure | None = None,
    exit_reason: ConversationEndReason | None = None,
) -> Transcript:
    """The transcript record of one conversation of ``case`` in this run (always sample 1)."""
    record = case.test_case
    sut = manifest.primary_sut.output_key
    return Transcript(
        transcript_id=f"{manifest.run_id}:{sut}:{record.test_case_id}:1",
        test_case_id=record.test_case_id,
        test_case_version=record.version,
        run_id=manifest.run_id,
        system_under_test=sut,
        sample=1,
        turns=turns,
        sim_failure=sim_failure,
        exit_reason=exit_reason,
        produced_by=produced_by(),
    )


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge overlay into base, preserving base keys not in overlay.

    - overlay values override base values
    - overlay None values are skipped (don't wipe base)
    - nested dicts are merged recursively
    - base is not mutated
    """
    merged = dict(base)
    for k, v in overlay.items():
        if v is None:
            continue
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
