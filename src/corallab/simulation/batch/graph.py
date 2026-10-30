"""Batch-level Burr graphs for parallel judge and sim execution.

Evaluation batch (``BatchState``):
    load → judge (MapStates) → write

Sim batch (``SimBatchState``):
    simulate (MapStates) → write

State schemas live in ``corallab.schemas.eval.graph_state``.
"""

from __future__ import annotations

from burr.core import default
from burr.core.graph import GraphBuilder
from burr.core.parallelism import RunnableGraph

from corallab.simulation.batch.load_judge_inputs import LoadJudgeInputs
from corallab.simulation.batch.parallel_judge import ParallelJudge
from corallab.simulation.batch.write_results import WriteResults


def build_eval_batch_graph(per_case_graph: RunnableGraph) -> GraphBuilder:
    """Build the batch-level orchestration graph for judge/run commands."""
    return (
        GraphBuilder()
        .with_actions(
            load=LoadJudgeInputs(),
            judge=ParallelJudge(per_case_graph),
            write=WriteResults(),
        )
        .with_transitions(
            ("load", "judge", default),
            ("judge", "write", default),
        )
    )


def build_sim_batch_graph(per_case_graph: RunnableGraph) -> GraphBuilder:
    """Build the batch-level orchestration graph for the sim command.

    Cases are already loaded and sliced by ``run_batch_sim``. This graph
    starts at ``simulate``.
    """
    from corallab.simulation.batch.parallel_sim import ParallelSim
    from corallab.simulation.batch.write_sim_results import WriteSimResults

    return (
        GraphBuilder()
        .with_actions(
            simulate=ParallelSim(per_case_graph),
            write=WriteSimResults(),
        )
        .with_transitions(
            ("simulate", "write", default),
        )
    )
