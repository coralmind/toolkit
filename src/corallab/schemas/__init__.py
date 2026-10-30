"""Pydantic data models shared across CoralMind toolkit stages.

Catalog entries come from `coralmap`.
"""

from corallab.schemas.core.kl import KlPin
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.conversation_signal import ConversationSignal
from corallab.schemas.eval.graph_state import (
    BatchState,
    PerTestCaseState,
    SimBatchState,
    SimulationLoopState,
)
from corallab.schemas.eval.persona import Persona, PersonaStates, PersonaTraits
from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import (
    JudgeConfig,
    ModelCallConfig,
    RunManifest,
    SimulationConfig,
)
from corallab.schemas.eval.test_case_record import TestCaseRecord
from corallab.schemas.eval.trace import SimulationResult, SimulationTrace
from corallab.schemas.eval.transcript import Transcript
from corallab.schemas.judge.eval_result import (
    EvalResult,
    EvalStatus,
    JudgePassInput,
    JudgePassResult,
)

__all__ = [
    "BatchState",
    "ConversationSignal",
    "EvalResult",
    "EvalStatus",
    "JudgeConfig",
    "JudgePassInput",
    "JudgePassResult",
    "KlPin",
    "ModelCallConfig",
    "PerTestCaseState",
    "Persona",
    "PersonaRecord",
    "PersonaStates",
    "PersonaTraits",
    "RunCase",
    "RunManifest",
    "SimBatchState",
    "SimulationConfig",
    "SimulationLoopState",
    "SimulationResult",
    "SimulationTrace",
    "TestCaseRecord",
    "Transcript",
    "Turn",
]
