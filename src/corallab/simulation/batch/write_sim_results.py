"""WriteSimResults action — writes sim output Transcript records to JSONL."""

from __future__ import annotations

import logging
from typing import Any

from burr.core import Action, State

logger = logging.getLogger(__name__)


class WriteSimResults(Action):
    """Write the sim output Transcript records to a JSONL file."""

    @property
    def reads(self) -> list[str]:
        # sim_outputs: from ParallelSim.reduce() — Transcript records
        # output_path: from initial state (CLI) — destination JSONL file path
        return ["sim_outputs", "output_path"]

    @property
    def writes(self) -> list[str]:
        # results_written: count of Transcript records written, for logging/diagnostics
        return ["results_written"]

    def run(self, state: State, **kwargs: Any) -> dict:
        from pathlib import Path

        sim_outputs = state["sim_outputs"]
        output_path = Path(state["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            for transcript in sim_outputs:
                f.write(transcript.model_dump_json() + "\n")

        logger.info("Wrote %d sim results to %s", len(sim_outputs), output_path)
        return {"results_written": len(sim_outputs)}

    def update(self, result: dict, state: State) -> State:
        return state.update(**result)

    def is_async(self) -> bool:
        return False
