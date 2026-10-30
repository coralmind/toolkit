"""Tracing hook — bridges Burr action tracking to viewer-compatible .trace.jsonl.

Implements Burr PreRunStepHook and PostRunStepHook to capture action-level events
and write them in the format the results viewer expects.

Two trace mechanisms:
1. Burr lifecycle hooks (automatic) — action start/end events for the batch graph
2. Explicit calls — log_llm_call() called from pipeline code (LLMClient, etc.)
   via the active hook context variable

Usage:
    hook = TracingHook(trace_path, eval_id="eval-xxx")
    token = set_active_hook(hook)        # make available to pipeline code
    builder.with_hooks(hook)              # register with Burr for action events
    ...
    reset_active_hook(token)              # cleanup after run
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from burr.lifecycle.base import PostRunStepHook, PreRunStepHook

logger = logging.getLogger(__name__)

_active_hook: contextvars.ContextVar[TracingHook | None] = contextvars.ContextVar(
    "trace_hook", default=None
)

# ContextVar so concurrent MapStates tasks do not share one test-case id.
_active_test_case_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_test_case_id", default=None
)


def set_active_hook(hook: TracingHook) -> contextvars.Token:
    """Set the active tracing hook for the current context. Returns a reset token."""
    return _active_hook.set(hook)


def get_active_hook() -> TracingHook | None:
    """Get the active tracing hook, or None if tracing is not enabled."""
    return _active_hook.get()


def reset_active_hook(token: contextvars.Token) -> None:
    """Reset the active hook to its previous value using the token from set_active_hook."""
    _active_hook.reset(token)


def set_active_test_case_id(test_case_id: str | None) -> contextvars.Token:
    """Set the ambient test_case_id for the current context. Returns a reset token."""
    return _active_test_case_id.set(test_case_id)


def get_active_test_case_id() -> str | None:
    """Return the ambient test_case_id set by the Burr hook or caller, if any.

    Used by ``LLMClient.complete()`` as the fallback when no explicit
    ``trace_test_case_id`` is passed. Judge callers that thread the id through
    function signatures override this value — the ambient is only consulted
    when a caller hasn't set one.
    """
    return _active_test_case_id.get()


def reset_active_test_case_id(token: contextvars.Token) -> None:
    """Reset the ambient test_case_id to its previous value."""
    _active_test_case_id.reset(token)


class TracingHook(PreRunStepHook, PostRunStepHook):
    """Burr lifecycle hook that writes viewer-compatible .trace.jsonl entries.

    Captures action execution events and writes them as JSONL entries
    compatible with the results viewer's TraceLoader.

    Registers with Burr via ApplicationBuilder.with_hooks() for automatic
    action start/end events. Also exposes log_llm_call() for explicit
    tracing from pipeline code.
    """

    def __init__(self, trace_path: Path, eval_id: str) -> None:
        self._trace_path = Path(trace_path)
        self._eval_id = eval_id
        self._seq = 0
        self._lock = threading.Lock()
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Append a single trace entry to the JSONL file."""
        try:
            with self._lock, open(self._trace_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to write trace entry: %s", e)

    # -- Burr lifecycle hooks (called automatically by Burr) ------------------

    def pre_run_step(
        self,
        *,
        state: Any,
        action: Any,
        **future_kwargs: Any,
    ) -> None:
        """Record action start as a pipeline event.

        Also propagates ``state["test_case_id"]`` into the ambient
        ``_active_test_case_id`` ContextVar so LLM calls inside the action
        pick it up without signature plumbing. Each asyncio Task (i.e. each
        MapStates fan-out) gets its own ContextVar copy — no cross-task
        contamination. ``state`` objects that don't carry test_case_id
        (e.g., run-level setup actions) are skipped silently.
        """
        try:
            tc_id = state.get("test_case_id", None)
        except Exception:
            tc_id = None
        if tc_id is not None:
            _active_test_case_id.set(tc_id)

        self._write_entry(
            {
                "type": "event",
                "seq": self._next_seq(),
                "eval_id": self._eval_id,
                "role": f"action_{action.name}",
                "turn": 0,
                "detail": {"phase": "start", "action": action.name},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def post_run_step(
        self,
        *,
        state: Any,
        action: Any,
        result: dict | None,
        sequence_id: int,
        exception: Exception | None,
        **future_kwargs: Any,
    ) -> None:
        """Record action completion."""
        self._write_entry(
            {
                "type": "event",
                "seq": self._next_seq(),
                "eval_id": self._eval_id,
                "role": f"action_{action.name}",
                "turn": 0,
                "detail": {
                    "phase": "end",
                    "action": action.name,
                    "sequence_id": sequence_id,
                    "error": str(exception) if exception else None,
                },
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    # -- Explicit logging (called from pipeline code) -------------------------

    def log_llm_call(
        self,
        *,
        role: str,
        turn: int | None = None,
        model: str,
        system: str | None = None,
        prompt: str,
        response: str,
        latency_ms: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int | None = None,
        endpoint_name: str | None = None,
        resolved_litellm_model: str | None = None,
        test_case_id: str | None = None,
    ) -> None:
        """Record an LLM call.

        ``endpoint_name`` and ``resolved_litellm_model`` describe the
        registry entry that served the call (see ``corallab.llm.router``).
        ``LLMClient`` always sets both. They stay optional for callers that
        record a call without a registry lookup.

        ``test_case_id`` is the originating test case id when the call is
        part of a per-test-case flow (judge calls, sim turns). ``None`` for
        run-level calls that aren't attributable to one case. Trace
        consumers rely on this field to split rows per test case in
        multi-case batches.
        """
        self._write_entry(
            {
                "type": "llm_call",
                "seq": self._next_seq(),
                "eval_id": self._eval_id,
                "test_case_id": test_case_id,
                "role": role,
                "turn": turn,
                "model": model,
                "system": system,
                "prompt": prompt,
                "response": response,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "endpoint_name": endpoint_name,
                "resolved_litellm_model": resolved_litellm_model,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    @property
    def eval_id(self) -> str:
        return self._eval_id

    @eval_id.setter
    def eval_id(self, value: str) -> None:
        self._eval_id = value
