"""WriteResults action — writes all evaluation results to JSONL output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import BatchState

logger = logging.getLogger(__name__)


class WriteResults(Action):
    """Write all results to JSONL output file."""

    @property
    def reads(self) -> list[str]:
        # results: from ParallelJudge.reduce() — list of EvalResult to serialize
        # output_path: from initial state (CLI) — destination JSONL file path
        return ["results", "output_path"]

    @property
    def writes(self) -> list[str]:
        # results_written: count of results written, for logging/diagnostics
        return ["results_written"]

    def run(self, state: State[BatchState], **kwargs: Any) -> dict:
        from pathlib import Path

        results = state["results"]
        output_path = Path(state["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")

        logger.info("Wrote %d results to %s", len(results), output_path)
        return {"results_written": len(results)}

    def update(self, result: dict, state: State[BatchState]) -> State[BatchState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return False
