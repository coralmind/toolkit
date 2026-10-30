"""LoadJudgeInputs action — loads test case records and transcripts, joined by test case."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from burr.core import Action, State

if TYPE_CHECKING:
    from corallab.schemas.eval.graph_state import BatchState
    from corallab.schemas.eval.transcript import Transcript

logger = logging.getLogger(__name__)


def _read_transcripts(path: Path) -> list[Transcript]:
    from corallab.schemas.eval.transcript import Transcript

    lines = path.read_text(encoding="utf-8").splitlines()
    return [Transcript.model_validate_json(line) for line in lines if line.strip()]


class LoadJudgeInputs(Action):
    """Load the test case records and transcripts; an unplaced transcript fails the run."""

    @property
    def reads(self) -> list[str]:
        return ["input_path", "transcripts_path", "manifest"]

    @property
    def writes(self) -> list[str]:
        return ["run_cases", "transcripts"]

    async def run(self, state: State[BatchState], **kwargs: Any) -> dict:  # type: ignore[override]
        from corallab.kl import load_library
        from corallab.simulation.data.test_case_records import load_test_case_records

        manifest = state["manifest"]
        cases = load_test_case_records(
            state["input_path"],
            manifest.persona_records_path,
            load_library(),
            evaluation_level=manifest.evaluation_level,
        )
        by_id = {c.test_case_id: c for c in cases}

        path = Path(state["transcripts_path"])
        transcripts = _read_transcripts(path)
        suts = {m.output_key for m in manifest.system_under_test}
        problems: list[str] = []
        for t in transcripts:
            if t.system_under_test not in suts:
                problems.append(
                    f"{t.transcript_id}: system under test {t.system_under_test} is not in the "
                    f"manifest's system_under_test ({sorted(suts)})"
                )
            case = by_id.get(t.test_case_id)
            if case is None:
                problems.append(
                    f"{t.transcript_id}: test case {t.test_case_id} is not in the test case records"
                )
            elif case.test_case.version != t.test_case_version:
                problems.append(
                    f"{t.transcript_id}: simulated version {t.test_case_version}, "
                    f"the test case records hold {case.test_case.version}"
                )
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
        logger.info("Loaded %d transcript(s) from %s", len(transcripts), path)
        return {"run_cases": cases, "transcripts": transcripts}

    def update(self, result: dict, state: State[BatchState]) -> State[BatchState]:
        return state.update(**result)

    def is_async(self) -> bool:
        return True
