"""The judge dispatch REGISTRY: one entry, the Knowledge Library two-stage judge.

RunSingleJudgePass looks a pass's ``judge_type`` up here and awaits
``spec.execute(state, pass_input)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from corallab.judging.judges.spec import JudgeSpec
from corallab.schemas.judge.eval_result import JudgePassResult

if TYPE_CHECKING:
    from burr.core import State

    from corallab.schemas.judge.eval_result import JudgePassInput

logger = logging.getLogger(__name__)


class KlTwoStageJudgeSpec(JudgeSpec):
    """KL two-stage judge: behavior identification, then pathway assessment.

    Options (group size, concurrency, pathways per call, prompt version, pricing)
    come from ``JudgeConfig.options``.
    """

    judge_type: ClassVar[str] = "kl_two_stage"

    async def execute(
        self,
        state: State,
        pass_input: JudgePassInput,
    ) -> JudgePassResult:
        # Import at call time so tests can patch kl_two_stage.run_kl_two_stage_judge.
        from corallab.judging.judges import kl_two_stage as kl_mod

        case = state["run_case"]
        transcript = state["transcript"]
        run = await kl_mod.run_kl_two_stage_judge(
            case=case,
            turns=transcript.turns or [],
            manifest=state["manifest"],
            model=pass_input.model,
            options=pass_input.options,
        )
        if run.run_status == "judge_error":
            logger.warning(
                "corallab judge run failed for test_case_id=%s model=%s: %s",
                case.test_case_id,
                pass_input.model.model_id,
                run.run_error_detail,
            )
        return JudgePassResult(
            judge_type=self.judge_type,
            coralmind_run=run,
            prompt="",
        )


REGISTRY: dict[str, JudgeSpec] = {"kl_two_stage": KlTwoStageJudgeSpec()}
