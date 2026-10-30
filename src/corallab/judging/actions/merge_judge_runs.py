"""MergeJudgeRuns action — collects judge pass results and builds EvalResult."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import PerTestCaseState

logger = logging.getLogger(__name__)


class MergeJudgeRuns(Action):
    """Collect the judge pass results and build the EvalResult."""

    @property
    def reads(self) -> list[str]:
        # judge_pass_results: from FanOutJudges.reduce() — all completed JudgePassResults
        # run_case, transcript: from ParallelJudge.states() — identity and the judged turns
        # manifest: from initial state (CLI) — debug flag, model IDs for EvalResult
        # evaluation_id: from ParallelJudge.states() — unique run identifier
        # evaluation_level: from ParallelJudge.states() — recorded on the EvalResult
        return [
            "judge_pass_results",
            "run_case",
            "transcript",
            "manifest",
            "evaluation_id",
            "evaluation_level",
        ]

    @property
    def writes(self) -> list[str]:
        # final_result: the EvalResult, collected by ParallelJudge.reduce()
        return ["final_result"]

    def run(self, state: State[PerTestCaseState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.code_version import produced_by
        from corallab.schemas.judge.eval_result import (
            EvalResult,
            EvalStatus,
            JudgePassResult,
        )

        pass_results: list[JudgePassResult] = state["judge_pass_results"]
        case = state["run_case"]
        transcript = state["transcript"]
        record = case.test_case
        manifest = state["manifest"]
        evaluation_id = state["evaluation_id"]
        # LoadJudgeInputs rejects a transcript whose system_under_test matches no entry.
        evaluated = next(
            m for m in manifest.system_under_test if m.output_key == transcript.system_under_test
        )

        coralmind_runs = [pr for pr in pass_results if pr.coralmind_run is not None]
        for extra in coralmind_runs[1:]:
            dropped = extra.coralmind_run
            logger.info(
                "Multiple coralmind passes on %s; keeping first, dropping model=%s",
                record.test_case_id,
                dropped.run_meta.model if dropped is not None else "?",
            )
        coralmind_run = coralmind_runs[0].coralmind_run if coralmind_runs else None
        failed = coralmind_run is None or coralmind_run.run_status == "judge_error"
        status = EvalStatus.JUDGE_ERROR.value if failed else EvalStatus.SUCCESS.value

        judge_prompt_snapshot = (
            {
                (
                    pr.coralmind_run.run_meta.model
                    if pr.coralmind_run is not None
                    else pr.judge_type
                ): pr.prompt
                for pr in pass_results
                if pr.prompt
            }
            if manifest.debug
            else None
        )
        judge_model_id = (
            coralmind_run.run_meta.model
            if coralmind_run is not None
            else manifest.judges[0].model.model_id
        )
        logger.info(
            "Evaluation complete: test_case_id=%s evaluation_id=%s status=%s",
            record.test_case_id,
            evaluation_id,
            status,
        )
        final_result = EvalResult(
            test_case_id=record.test_case_id,
            test_case_version=record.version,
            test_case_display_name=record.display_name,
            test_case_description=record.description,
            transcript_id=transcript.transcript_id,
            run_id=transcript.run_id,
            system_under_test=transcript.system_under_test,
            evaluation_level=state["evaluation_level"],
            evaluation_id=evaluation_id,
            evaluated_model_id=evaluated.model_id,
            evaluated_at=datetime.now(UTC).isoformat(),
            judge_model_id=judge_model_id,
            status=status,
            produced_by=produced_by(),
            conversation_transcript=transcript.turns,
            coralmind_run=coralmind_run,
            judge_prompt_snapshot=judge_prompt_snapshot,
        )
        return {"final_result": final_result}

    def update(self, result: dict, state: State[PerTestCaseState]) -> State[PerTestCaseState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return False
