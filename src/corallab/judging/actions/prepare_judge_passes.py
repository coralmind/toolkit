"""PrepareJudgePasses action — resolves the manifest's judge passes for one transcript."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import PerTestCaseState

logger = logging.getLogger(__name__)


class PrepareJudgePasses(Action):
    """Resolve manifest judge config into a JudgePassInput per enabled pass."""

    @property
    def reads(self) -> list[str]:
        # transcript: from ParallelJudge.states() — the conversation to judge
        # manifest: from initial state (CLI) — judges and the model each one calls
        return ["transcript", "manifest"]

    @property
    def writes(self) -> list[str]:
        # judge_pass_inputs: list of JudgePassInput (model, options, etc.)
        #   consumed by FanOutJudges to spawn parallel sub-graphs
        return ["judge_pass_inputs"]

    def run(self, state: State[PerTestCaseState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.judging.judges import REGISTRY
        from corallab.schemas.judge.eval_result import JudgePassInput

        transcript = state["transcript"]
        manifest = state["manifest"]

        if transcript is None or transcript.turns is None:
            msg = "PrepareJudgePasses needs a transcript with turns; run `corallab sim` first."
            raise ValueError(msg)

        judges = manifest.judges

        inputs: list[JudgePassInput] = []
        for judge_cfg in judges:
            if not judge_cfg.enabled:
                continue

            judge_type = judge_cfg.judge_type
            spec = REGISTRY.get(judge_type)
            if spec is None:
                logger.warning(
                    "Unknown judge_type %r for model %r — skipping",
                    judge_type,
                    judge_cfg.model.model_id,
                )
                continue

            inputs.append(
                JudgePassInput(
                    judge_type=judge_type,
                    prompt="",
                    model=judge_cfg.model,
                    options=dict(judge_cfg.options),
                )
            )

        logger.debug(
            "Prepared %d judge passes for test_case_id=%s: %s",
            len(inputs),
            transcript.test_case_id,
            [i.model.model_id for i in inputs],
        )
        return {"judge_pass_inputs": inputs}

    def update(self, result: dict, state: State[PerTestCaseState]) -> State[PerTestCaseState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return False
