"""Evaluation result schemas: the per-transcript result and the judge fan-out wrappers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from corallab.schemas.core.code_version import ProducedBy
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_manifest import ModelCallConfig
from corallab.schemas.judge.coralmind_result import CoralMindRun

# ---------------------------------------------------------------------------
# Evaluation result enums
# ---------------------------------------------------------------------------


class EvalStatus(StrEnum):
    """Pipeline execution status."""

    SUCCESS = "success"
    JUDGE_ERROR = "judge_error"  # Judge call failed or returned unparseable output


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """The judge's result for one transcript: which test case, which conversation, what it found."""

    # --- Identity ---
    test_case_id: str = Field(description="Test case the transcript was simulated from")
    test_case_version: str = Field(description="Version of that test case")
    test_case_display_name: str | None = Field(
        default=None, description="Short human-readable name for dashboard display"
    )
    test_case_description: str | None = Field(
        default=None, description="Longer description for detail view"
    )
    transcript_id: str = Field(description="Transcript this result judges")
    run_id: str = Field(description="Run manifest run_id that produced the transcript")
    system_under_test: str = Field(description="Output key of the system under test that answered")
    evaluation_level: int = Field(description="Evaluation level (1-3)")

    # --- Evaluation metadata ---
    evaluation_id: str = Field(description="Unique ID of this judge evaluation")
    evaluated_model_id: str = Field(description="Model id of the system under test")
    evaluated_at: str = Field(description="ISO 8601 timestamp of evaluation")
    judge_model_id: str = Field(description="Model used as judge")
    status: str = Field(default="success", description="Pipeline status: success or judge_error")
    produced_by: ProducedBy | None = Field(
        default=None,
        description=(
            "Toolkit and Knowledge Library versions that wrote this result. "
            "Null on a record written before this was recorded"
        ),
    )

    # --- Transcript ---
    conversation_transcript: list[Turn] | None = Field(
        default=None, description="The judged conversation's turns"
    )

    # --- Judge result ---
    coralmind_run: CoralMindRun | None = Field(
        default=None,
        description=(
            "Two-stage judge result (behavior observations, pathway assessments, run "
            "metadata). None if the judge did not run"
        ),
    )

    # --- Burr orchestration ---
    burr_project: str | None = Field(
        default=None, description="Burr tracking project name for cross-linking to Burr UI"
    )

    # --- Debug ---
    judge_prompt_snapshot: dict[str, str] | None = Field(
        default=None, description="Debug: Captured judge prompts for inspection"
    )


# ---------------------------------------------------------------------------
# Internal pipeline intermediate types
# ---------------------------------------------------------------------------


class JudgePassInput(BaseModel):
    """Resolved per-pass input produced by PrepareJudgePasses, consumed by RunSingleJudgePass.

    Carries the judge's :class:`ModelCallConfig` and options for this transcript.
    """

    judge_type: str = Field(description="REGISTRY key of the judge to run")
    prompt: str = Field(description="Assembled prompt text")
    model: ModelCallConfig = Field(description="Model this pass calls, from JudgeConfig.model")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Judge-specific options copied from JudgeConfig.options",
    )


class JudgePassResult(BaseModel):
    """One judge pass result, collected by FanOutJudges.reduce()."""

    judge_type: str = Field(description="'kl_two_stage'")
    coralmind_run: CoralMindRun | None = Field(
        default=None, description="Populated for kl_two_stage passes"
    )
    prompt: str = Field(default="", description="Assembled prompt text (for debug snapshots)")
