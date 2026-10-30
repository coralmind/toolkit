"""``corallab check`` — run the fidelity check on existing transcripts.

Input is the Transcript JSONL from ``corallab sim``, and the manifest's test case records
(``test_case_records_path``) and persona directory (``persona_records_path``). Writes
``<stem>-check.jsonl`` beside the transcripts: one ``FidelityCheckRecord`` per transcript, read
by the results viewer. The transcripts themselves are not changed.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from corallab.cli._helpers import (
    console,
    err_console,
    init_logging,
    print_token_summary,
    resolve_run_manifest,
    transcript_run_dir,
)

if TYPE_CHECKING:
    from corallab.schemas.eval.run_manifest import RunManifest

logger = logging.getLogger(__name__)


def check(
    ctx: typer.Context,
    run_manifest: Annotated[str, typer.Option(help="Path to run manifest YAML")],
    transcripts: Annotated[
        str, typer.Option(help="Path to the Transcript JSONL written by `corallab sim`")
    ],
) -> None:
    """Check whether each simulated person did what its test case required."""
    from corallab.fidelity.elicitation import prompt_problem

    manifest = resolve_run_manifest(run_manifest)
    if manifest.evaluation_level < 3:
        err_console.print("[bold red]Error:[/] scripted levels (L1, L2) have no fidelity check")
        raise typer.Exit(1)
    problem = prompt_problem(manifest)
    if problem:
        err_console.print(f"[bold red]Error:[/] {problem}")
        raise typer.Exit(1)
    init_logging(ctx, manifest, out_dir=transcript_run_dir(transcripts))
    source = Path(transcripts)
    out_path = _execute_check(manifest, source, source.parent, source.stem.removesuffix("-sim"))
    console.print(f"[bold green]Done.[/] Verdicts written to [cyan]{out_path}[/]")


def _simulated_records(manifest: RunManifest, transcripts: Path) -> str | Path:
    """The test case records the transcripts were simulated from: the run's ``inputs/`` copy,
    else the manifest's path."""
    run_dir = transcript_run_dir(transcripts)
    if run_dir is not None:
        copied = run_dir / "inputs" / Path(manifest.test_case_records_path).name
        if copied.is_file():
            return copied
    return manifest.test_case_records_path


def _execute_check(manifest: RunManifest, transcripts: Path, out_dir: Path, run_stem: str) -> Path:
    """Check every transcript in ``transcripts``. Returns the output path."""
    from corallab.fidelity.elicitation import ElicitationFidelityChecker
    from corallab.kl import load_library
    from corallab.llm_client import (
        TokenAccumulator,
        llm_client,
        reset_token_accumulator,
        set_token_accumulator,
    )
    from corallab.schemas.eval.transcript import Transcript
    from corallab.simulation.data.test_case_records import load_test_case_records
    from corallab.simulation.hooks import TracingHook, reset_active_hook, set_active_hook

    warning = _simulator_checks_itself(manifest)
    if warning:
        logger.warning(warning)
        err_console.print(f"[bold yellow]Warning:[/] {warning}")
    out_path = out_dir / f"{run_stem}-check.jsonl"
    cases = load_test_case_records(
        _simulated_records(manifest, transcripts),
        manifest.persona_records_path,
        load_library(),
        evaluation_level=manifest.evaluation_level,
    )
    by_id = {c.test_case_id: c.test_case for c in cases}
    records = [
        Transcript.model_validate_json(line)
        for line in transcripts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checker = ElicitationFidelityChecker(run_config=manifest, llm_client=llm_client)
    hook_token = set_active_hook(
        TracingHook(out_dir / f"{run_stem}-check.trace.jsonl", eval_id=f"{run_stem}-check")
    )
    accumulator = TokenAccumulator()
    accumulator_token = set_token_accumulator(accumulator)
    try:
        results = asyncio.run(
            _check_transcripts(checker, by_id, records, manifest.simulation.batch_size)
        )
    except Exception as exc:
        logger.error("Fatal error in fidelity check: %s", exc, exc_info=True)
        raise typer.Exit(1) from exc
    finally:
        reset_active_hook(hook_token)
        reset_token_accumulator(accumulator_token)
    out_path.write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    counts: dict[str, int] = {}
    for result in results:
        status = result.fidelity.status if result.fidelity else "not checked"
        counts[status] = counts.get(status, 0) + 1
    console.print(
        f"[bold green]Fidelity check complete.[/] [cyan]{len(results)}[/] transcript(s): "
        + ", ".join(f"{status} {n}" for status, n in counts.items())
    )
    print_token_summary(accumulator, "check")
    return out_path


async def _check_transcripts(
    checker: Any, by_id: dict[str, Any], transcripts: list[Any], concurrency: int
) -> list[Any]:
    """One record per transcript with turns, ``concurrency`` checks at a time, in file order."""
    from corallab.code_version import produced_by
    from corallab.schemas.eval.transcript import FidelityCheckRecord

    gate = asyncio.Semaphore(concurrency)

    async def one(t: Any) -> FidelityCheckRecord | None:
        if t.turns is None:
            return None
        case = by_id.get(t.test_case_id)
        if case is None:
            raise ValueError(f"{t.transcript_id}: test case {t.test_case_id} is not in the run")
        async with gate:
            verdict = await checker.check(
                turns=t.turns, test_case=case, test_case_id=t.test_case_id
            )
        return FidelityCheckRecord(
            transcript_id=t.transcript_id,
            test_case_id=t.test_case_id,
            system_under_test=t.system_under_test,
            fidelity=verdict,
            error=None if verdict is not None else "the check could not be completed; see the log",
            produced_by=produced_by(),
        )

    return [r for r in await asyncio.gather(*(one(t) for t in transcripts)) if r]


def _simulator_checks_itself(manifest: RunManifest) -> str | None:
    """A warning when the checker is the model that played the person."""
    fidelity = manifest.simulation.elicitation_fidelity
    person = manifest.simulation.person_response_model
    checker = fidelity.model or person
    if checker is None or person is None or checker.model_id != person.model_id:
        return None
    how = (
        "no elicitation_fidelity.model is set"
        if fidelity.model is None
        else "elicitation_fidelity.model names it"
    )
    return f"the checker is the simulator's own model, {checker.model_id} ({how})"
