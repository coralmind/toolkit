"""The JudgeSpec ABC: the single extension point for a judge type.

Concrete subclasses and the dispatch ``REGISTRY`` live in ``registry.py``;
``judges/__init__.py`` re-exports both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from burr.core import State

    from corallab.schemas.judge.eval_result import JudgePassInput, JudgePassResult


class JudgeSpec(ABC):
    """A judge type. Each subclass owns its whole execution through ``execute``.

    ``execute(state, pass_input)`` takes the per-test-case state (``run_case``,
    ``transcript``, ``manifest``) and the pass's model, and returns a completed
    ``JudgePassResult``; a subclass may fire one LLM call or fan out internally.
    """

    # Dispatch key for REGISTRY, and the value stamped on JudgePassResult.
    judge_type: ClassVar[str]

    @abstractmethod
    async def execute(
        self,
        state: State,
        pass_input: JudgePassInput,
    ) -> JudgePassResult:
        """Run this judge to completion and return a JudgePassResult."""
