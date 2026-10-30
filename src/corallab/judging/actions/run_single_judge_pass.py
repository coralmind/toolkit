"""RunSingleJudgePass action — thin dispatcher that awaits spec.execute()."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import PerTestCaseState

logger = logging.getLogger(__name__)


class RunSingleJudgePass(Action):
    """Execute a single judge pass by delegating to its JudgeSpec.

    Each ``JudgeSpec`` subclass owns its execution contract (ABC in
    ``judging/judges/spec.py``; concrete subclasses + dispatch
    ``REGISTRY`` in ``judging/judges/registry.py``). This action is a
    thin dispatcher: look up the spec by ``judge_type``, await
    ``spec.execute(...)``, write the result. Adding a new judge type never
    requires touching this action.
    """

    @property
    def reads(self) -> list[str]:
        # judge_pass_input: from FanOutJudges.states() — single JudgePassInput.
        # manifest, run_case, transcript, evaluation_level: required by every spec's execute().
        return ["judge_pass_input", "manifest", "run_case", "transcript", "evaluation_level"]

    @property
    def writes(self) -> list[str]:
        # judge_pass_result: completed JudgePassResult collected by FanOutJudges.reduce().
        return ["judge_pass_result"]

    async def run(self, state: State[PerTestCaseState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.judging.judges import REGISTRY

        cfg = state["judge_pass_input"]
        spec = REGISTRY.get(cfg.judge_type)
        if spec is None:
            msg = f"Unknown judge_type {cfg.judge_type!r}"
            raise ValueError(msg)

        result = await spec.execute(state, cfg)
        return {"judge_pass_result": result}

    def update(self, result: dict, state: State[PerTestCaseState]) -> State[PerTestCaseState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
