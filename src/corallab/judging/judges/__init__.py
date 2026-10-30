"""Judge registry package — re-exports the JudgeSpec ABC and the one-entry dispatch REGISTRY.

The ABC lives in ``spec.py``; the concrete subclass and the
``REGISTRY`` dict live in ``registry.py``. Registry imports stay lazy so ``spec`` can be
imported without pulling in the judge implementation.
"""

from __future__ import annotations

from typing import Any

from corallab.judging.judges.spec import JudgeSpec

__all__ = ["REGISTRY", "JudgeSpec", "KlTwoStageJudgeSpec"]


def __getattr__(name: str) -> Any:
    if name in {"REGISTRY", "KlTwoStageJudgeSpec"}:
        from corallab.judging.judges import registry as _registry

        return getattr(_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
