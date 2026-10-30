"""Burr Application runners for CLI commands.

Each function builds a Burr Application (graph + tracing + persistence + state)
and executes it. Graph topology lives in the per-phase ``graph.py`` modules
(``simulation.graph``, ``judging.graph``, ``simulation.batch.graph``);
this module only handles runtime wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corallab.schemas.eval.run_case import RunCase
    from corallab.schemas.eval.run_manifest import RunManifest
    from corallab.schemas.eval.transcript import Transcript
    from corallab.schemas.judge.eval_result import EvalResult

logger = logging.getLogger(__name__)


async def _build_persister(db_path: Path | None) -> Any:
    """Create and initialize an AsyncSQLitePersister if *db_path* is set."""
    if db_path is None:
        return None
    from burr.integrations.persisters.b_aiosqlite import AsyncSQLitePersister

    persister = await AsyncSQLitePersister.from_values(
        db_path=str(db_path),
        table_name="corallab_state",
    )
    await persister.initialize()
    return persister


async def _build_and_run(
    *,
    graph: Any,
    initial_state: dict[str, Any],
    typing_model: type,
    entrypoint: str,
    halt_after: list[str],
    trace_dir: Path,
    run_stem: str,
    db_path: Path | None,
    dry_run: bool = False,
) -> tuple[Any, Any]:
    """Build a Burr app with standard tracing + persistence, run it, return (state, accumulator)."""
    from burr.core import ApplicationBuilder
    from burr.integrations.pydantic import PydanticTypingSystem

    from corallab.llm_client import (
        TokenAccumulator,
        reset_token_accumulator,
        set_token_accumulator,
    )
    from corallab.simulation.hooks import (
        TracingHook,
        reset_active_hook,
        set_active_hook,
    )

    trace_path = trace_dir / f"{run_stem}.trace.jsonl"
    if dry_run:
        dry_trace_dir = trace_dir / "dry_run" / "traces"
        dry_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = dry_trace_dir / f"{run_stem}.trace.jsonl"
    hook = TracingHook(trace_path, eval_id=run_stem)
    hook_token = set_active_hook(hook)

    accumulator = TokenAccumulator()
    acc_token = set_token_accumulator(accumulator)

    builder = (
        ApplicationBuilder()
        .with_graph(graph)
        .with_state(**initial_state)
        .with_typing(PydanticTypingSystem(typing_model))
        .with_hooks(hook)
        .with_identifiers(app_id=run_stem, partition_key="corallab")
        .with_tracker("local", project="corallab")
        .with_entrypoint(entrypoint)
    )

    persister = await _build_persister(db_path)
    if persister is not None:
        builder = builder.with_state_persister(persister)

    burr_app = await builder.abuild()
    try:
        _, _, state = await burr_app.arun(halt_after=halt_after)
        return state, accumulator
    finally:
        reset_active_hook(hook_token)
        reset_token_accumulator(acc_token)
        if persister is not None:
            await persister.cleanup()


async def _run_sim_group(
    run_manifest: RunManifest,
    cases: list[RunCase],
    out_path: Path,
    db_path: Path | None,
    run_stem: str,
    dry_run: bool,
) -> tuple[list[Transcript], Any]:
    """Simulate ``cases`` together, writing transcripts to ``out_path``."""
    from corallab.schemas.eval.graph_state import SimBatchState
    from corallab.simulation.batch.graph import build_sim_batch_graph
    from corallab.simulation.graph import build_sim_subgraph

    per_case = build_sim_subgraph(run_manifest.evaluation_level)
    batch_graph = build_sim_batch_graph(per_case).build()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state, accumulator = await _build_and_run(
        graph=batch_graph,
        initial_state={
            "input_path": run_manifest.test_case_records_path,
            "manifest": run_manifest,
            "output_path": str(out_path),
            "run_cases": cases,
            "sim_outputs": [],
            "results_written": 0,
        },
        typing_model=SimBatchState,
        entrypoint="simulate",
        halt_after=["simulate"] if dry_run else ["write"],
        trace_dir=out_path.parent,
        run_stem=run_stem,
        db_path=db_path,
        dry_run=dry_run,
    )
    sim_outputs: list[Transcript] = state.get("sim_outputs", [])
    return sim_outputs, accumulator


async def run_batch_sim(
    run_manifest: RunManifest,
    input_path: str,
    out_path: Path,
    db_path: Path | None,
    run_stem: str,
    dry_run: bool = False,
) -> tuple[list[Transcript], list[RunCase], Any, list[Path]]:
    """Simulate every test case, ``simulation.batch_size`` cases at a time.

    Each batch is written to ``out_path.parent / batch-NNN / out_path.name`` and
    finishes before the next batch starts. ``out_path`` itself is not written.
    Within a batch every case is in flight together.

    When ``dry_run=True``, halts after ``simulate`` (before ``write``) so no
    output JSONL is produced — the LLM short-circuit in ``llm_client`` records
    prompt tokens without making API calls.

    Returns (transcripts, run_cases, token_accumulator, batch_paths): one Transcript
    record per conversation, the test cases the run loaded, and the JSONL path
    written for each batch. Those paths are where this call wrote, so an older
    ``batch-NNN`` left in the directory is not included.
    """
    from corallab.grouping import batch_directory, iter_groups
    from corallab.kl import load_library
    from corallab.llm_client import TokenAccumulator
    from corallab.simulation.data.test_case_records import load_test_case_records

    cases = load_test_case_records(
        input_path,
        run_manifest.persona_records_path,
        load_library(),
        evaluation_level=run_manifest.evaluation_level,
    )
    groups = list(iter_groups(cases, run_manifest.simulation.batch_size))
    collected: list[Transcript] = []
    written: list[Path] = []
    merged = TokenAccumulator()
    n_groups = len(groups)
    for index, group in groups:
        group_dir = batch_directory(out_path.parent, index)
        group_out = group_dir / out_path.name
        group_db = None if db_path is None else group_dir / Path(db_path).name
        logger.info(
            "Simulating batch %d/%d (%d test case(s)) → %s",
            index,
            n_groups,
            len(group),
            group_out,
        )
        outputs, accumulator = await _run_sim_group(
            run_manifest,
            group,
            group_out,
            group_db,
            f"{run_stem}-batch-{index:03d}",
            dry_run,
        )
        collected.extend(outputs)
        written.append(group_out)
        merged.merge(accumulator)
    return collected, cases, merged, written


async def run_batch_judge(
    run_manifest: RunManifest,
    transcripts_path: str,
    out_path: Path,
    db_path: Path | None,
    run_stem: str,
    dry_run: bool = False,
) -> tuple[list[EvalResult], Any]:
    """Execute the judge batch graph (judge-only, no sim/SUT) using Burr.

    The test case records are ``run_manifest.test_case_records_path``.
    ``transcripts_path`` is the Transcript JSONL written by ``corallab sim``,
    each line joined to its test case and persona.
    When dry_run=True, halts before WriteResults so no output JSONLs are written.
    Returns (results, token_accumulator).
    """
    from corallab.judging.graph import build_judge_subgraph
    from corallab.schemas.eval.graph_state import BatchState
    from corallab.simulation.batch.graph import build_eval_batch_graph

    per_case = build_judge_subgraph()
    batch_graph = build_eval_batch_graph(per_case).build()

    state, accumulator = await _build_and_run(
        graph=batch_graph,
        initial_state={
            "input_path": run_manifest.test_case_records_path,
            "transcripts_path": transcripts_path,
            "manifest": run_manifest,
            "output_path": str(out_path),
            "run_cases": [],
            "transcripts": [],
            "results": [],
            "results_written": 0,
        },
        typing_model=BatchState,
        entrypoint="load",
        halt_after=["judge"] if dry_run else ["write"],
        trace_dir=out_path.parent,
        run_stem=run_stem,
        db_path=db_path,
        dry_run=dry_run,
    )
    results: list[EvalResult] = state.get("results", [])
    return results, accumulator
