"""``corallab run-preview`` — visualize the evaluation Burr graphs.

Renders the actual batch graphs that the CLI commands execute:
- **sim**: ``load → simulate (MapStates) → write``  (L1/L2 or L3)
- **judge**: ``load → judge (MapStates) → write``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from corallab.cli._helpers import console, err_console, init_logging

logger = logging.getLogger(__name__)

VALID_GRAPHS = ("sim", "judge")


def run_preview(
    ctx: typer.Context,
    graph: Annotated[str, typer.Option(help="Graph(s) to render: sim, judge, all")] = "all",
    level: Annotated[int, typer.Option(help="Evaluation level for the sim graph (1-3)")] = 1,
    fmt: Annotated[str, typer.Option("--format", help="Output format: png, svg, dot, pdf")] = "png",
    output: Annotated[str, typer.Option(help="Output directory")] = "results/graphs",
    view: Annotated[
        bool, typer.Option("--view", help="Open rendered graph in system viewer")
    ] = False,
    conditions: Annotated[
        bool, typer.Option("--conditions", help="Include edge condition labels")
    ] = False,
) -> None:
    """Visualize the batch Burr graphs that the CLI actually executes."""
    init_logging(ctx)

    from burr.core import ApplicationBuilder
    from burr.integrations.pydantic import PydanticTypingSystem

    graphs_to_render = [g.strip().lower() for g in graph.split(",")]
    if "all" in graphs_to_render:
        graphs_to_render = list(VALID_GRAPHS)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in graphs_to_render:
        if name not in VALID_GRAPHS:
            err_console.print(
                f"[bold red]Unknown graph:[/] {name}. "
                f"Choose from: [cyan]{', '.join(VALID_GRAPHS)}[/]"
            )
            continue

        built_graph, state_model, initial_state, entrypoint = _build_graph(name, level)

        burr_app = (
            ApplicationBuilder()
            .with_graph(built_graph)
            .with_state(**initial_state)
            .with_typing(PydanticTypingSystem(state_model))
            .with_entrypoint(entrypoint)
            .build()
        )

        suffix = f"_l{level}" if name == "sim" else ""
        output_file = out_dir / f"graph_{name}{suffix}"
        burr_app.visualize(
            output_file_path=str(output_file),
            include_conditions=conditions,
            view=view,
            format=fmt,
        )
        console.print(f"  [cyan]{name}{suffix}[/]: {output_file}.{fmt}")

    console.print(f"Graphs written to [cyan]{out_dir}/[/]")


def _build_graph(name: str, level: int) -> tuple:
    """Return ``(graph, state_model, initial_state, entrypoint)`` for *name*."""
    if name == "sim":
        from corallab.schemas.eval.graph_state import SimBatchState
        from corallab.simulation.batch.graph import build_sim_batch_graph
        from corallab.simulation.graph import build_sim_subgraph

        per_case = build_sim_subgraph(level)
        built = build_sim_batch_graph(per_case).build()
        initial_state: dict[str, Any] = {
            "input_path": "",
            "manifest": None,
            "output_path": "",
            "run_cases": [],
            "sim_outputs": [],
            "results_written": 0,
        }
        return built, SimBatchState, initial_state, "simulate"

    from corallab.judging.graph import build_judge_subgraph
    from corallab.schemas.eval.graph_state import BatchState
    from corallab.simulation.batch.graph import build_eval_batch_graph

    per_case = build_judge_subgraph()
    built = build_eval_batch_graph(per_case).build()
    initial_state = {
        "input_path": "",
        "transcripts_path": "",
        "manifest": None,
        "output_path": "",
        "run_cases": [],
        "transcripts": [],
        "results": [],
        "results_written": 0,
    }
    return built, BatchState, initial_state, "load"
