"""Typed state schemas for Burr graph orchestration.

Defines Pydantic models documenting the state shape at each level:
- PerTestCaseState: state flowing through per-test-case graphs (L1, L2, L3)
- BatchState: state flowing through the batch orchestration graph (judge, run)
- SimBatchState: state flowing through the sim batch orchestration graph

These models serve three purposes:
1. Type annotations — State[PerTestCaseState] gives IDE autocomplete on state.data
2. Runtime validation — PydanticTypingSystem wires them into Burr's state.data property
3. Documentation — single source of truth for what each graph expects in state
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.trace import SimulationResult
from corallab.schemas.eval.transcript import Transcript
from corallab.schemas.judge.eval_result import (
    EvalResult,
    JudgePassInput,
    JudgePassResult,
)


class PerTestCaseState(BaseModel):
    """State for per-test-case evaluation graphs (L1/L2/L3).

    Sim flows through: SystemUnderTest/RunSimulation

    Judge flows through: PrepareJudgePasses → FanOutJudges (MapStates) →
    MergeJudgeRuns

    Fields are Optional/defaulted because not every action reads every field,
    and Burr initializes state incrementally.
    """

    run_case: RunCase | None = None
    transcript: Transcript | None = None
    manifest: RunManifest | None = None
    evaluation_id: str = ""
    evaluation_level: int = 1

    # Simulation outputs (L3 only; None for L1/L2)
    sim_result: SimulationResult | None = None

    # Judge fan-out intermediate state
    judge_pass_inputs: list[JudgePassInput] = Field(default_factory=list)
    judge_pass_results: list[JudgePassResult] = Field(default_factory=list)

    # Per-judge sub-graph state (set by FanOutJudges.states, read by RunSingleJudgePass)
    judge_pass_input: JudgePassInput | None = None
    judge_pass_result: JudgePassResult | None = None

    # Evaluation output
    final_result: EvalResult | None = None


class BatchState(BaseModel):
    """State for the batch orchestration graph (judge + run commands).

    Flows through: LoadJudgeInputs → ParallelJudge (MapStates) → WriteResults

    The ParallelJudge step fans out into per-case sub-graphs, each receiving
    a state that is a superset of BatchState + PerTestCaseState fields.
    """

    input_path: str = ""
    transcripts_path: str = ""
    manifest: RunManifest | None = None
    output_path: str = ""
    run_cases: list[RunCase] = Field(default_factory=list)
    transcripts: list[Transcript] = Field(default_factory=list)
    results: list[EvalResult] = Field(default_factory=list)
    results_written: int = 0


class SimulationLoopState(BaseModel):
    """State for the per-conversation simulation loop graph (L3).

    Flows through the conversation loop actions:
    SendOpening → CallSUT → CheckExit → GeneratePersonaMessage →
    (loop) → AssembleSimResult
    """

    # Config (set at init, read-only during loop)
    test_case_id: str = ""
    max_turns: int = 10

    # Conversation tracking. turn_number is the current exchange: the opening
    # user message is 1, and the assistant reply keeps that number. The next
    # user message increments it. round_count is assistant replies.
    conversation_history: list = Field(default_factory=list)
    turn_number: int = 0
    round_count: int = 0

    # Exit control
    should_exit: bool = False
    exit_reason: str = ""
    conversation_closing: bool = False
    scenario_enacted: bool = False

    # Output (set by AssembleSimResult)
    sim_result: Any = None


class SimBatchState(BaseModel):
    """State for the sim batch orchestration graph.

    Flows through: ParallelSim (MapStates) → WriteSimResults. ``run_cases`` is the
    slice ``run_batch_sim`` already loaded.

    Output is a JSONL of Transcript records, read by the judge command with the test case records.
    """

    input_path: str = ""
    manifest: RunManifest | None = None
    output_path: str = ""
    run_cases: list[RunCase] = Field(default_factory=list)
    sim_outputs: list[Transcript] = Field(default_factory=list)
    results_written: int = 0
