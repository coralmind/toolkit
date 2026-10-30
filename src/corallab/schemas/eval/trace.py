"""Trace schemas — simulation execution records and debugging superset.

trace.py embeds EvalResult so a single trace file provides full pipeline
traceability without joining separate files.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import ConversationEndReason
from corallab.schemas.judge.eval_result import EvalResult


class SimulationResult(BaseModel):
    """Completed simulation output: transcript and metadata."""

    test_case_id: str = Field(description="Test case this simulation was run for")
    run_config: RunManifest = Field(description="Run configuration used for this simulation")
    conversation_transcript: list[Turn] = Field(
        description="Complete conversation turns from the simulation"
    )
    exit_reason: ConversationEndReason | None = Field(
        default=None,
        description=(
            "Why the L3 loop stopped: 'max_turns', 'conversation_closing' or 'person_echo'. "
            "Copied onto the Transcript that sim writes"
        ),
    )
    total_turns: int = Field(
        default=0,
        description=(
            "Exchanges completed. Equals the number of assistant replies when the "
            "assistant spoke last"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional runtime metadata"
    )


class SimulationTrace(BaseModel):
    """Full debugging record — simulation execution + evaluation output.

    Embeds EvalResult so debugging the full pipeline requires only this file.
    """

    simulation_result: SimulationResult = Field(description="Complete simulation output")
    eval_result: EvalResult | None = Field(
        default=None, description="Evaluation result if judging was performed"
    )
