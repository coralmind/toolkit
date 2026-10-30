"""Judge-related Burr graph builders.

Provides graphs for single judge passes and the judge fan-out pipeline.

Single judge pass:
    run_judge

Judge-only pipeline (for pre-existing transcripts):
    prepare → fan_out → merge
"""

from __future__ import annotations

from burr.core import default
from burr.core.graph import GraphBuilder
from burr.core.parallelism import RunnableGraph

from corallab.judging.actions.fan_out_judges import FanOutJudges
from corallab.judging.actions.merge_judge_runs import MergeJudgeRuns
from corallab.judging.actions.prepare_judge_passes import PrepareJudgePasses
from corallab.judging.actions.run_single_judge_pass import RunSingleJudgePass


def build_single_judge_subgraph() -> RunnableGraph:
    """Build a RunnableGraph for a single judge pass (used inside FanOutJudges).

    Contains one action: RunSingleJudgePass.  Reads ``judge_pass_input``
    from state, writes ``judge_pass_result``.
    """
    graph = (
        GraphBuilder()
        .with_actions(
            run_judge=RunSingleJudgePass(),
        )
        .build()
    )
    return RunnableGraph(graph=graph, entrypoint="run_judge", halt_after=["run_judge"])


def build_judge_pipeline_graph() -> GraphBuilder:
    """Build the judge-only pipeline: prepare → fan_out → merge.

    Returns a GraphBuilder (caller must ``.build()``).  Expects state with
    a ``run_case`` and a ``transcript`` with turns.
    """
    single_judge = build_single_judge_subgraph()

    return (
        GraphBuilder()
        .with_actions(
            prepare_judges=PrepareJudgePasses(),
            fan_out_judges=FanOutJudges(single_judge),
            merge_judges=MergeJudgeRuns(),
        )
        .with_transitions(
            ("prepare_judges", "fan_out_judges", default),
            ("fan_out_judges", "merge_judges", default),
        )
    )


def build_judge_subgraph() -> RunnableGraph:
    """Build a per-test-case RunnableGraph for judge-only batch execution.

    Wraps :func:`build_judge_pipeline_graph` as a ``RunnableGraph`` suitable
    for use inside a ``MapStates`` fan-out.
    """
    graph = build_judge_pipeline_graph().build()
    return RunnableGraph(graph=graph, entrypoint="prepare_judges", halt_after=["merge_judges"])
