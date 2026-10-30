"""Verify TracingHook integrates correctly with Burr."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from corallab.simulation.hooks import (
    TracingHook,
    get_active_hook,
    reset_active_hook,
    set_active_hook,
)


class TestTracingHookBurrIntegration:
    def test_inherits_burr_abcs(self) -> None:
        from burr.lifecycle.base import PostRunStepHook, PreRunStepHook

        hook = TracingHook(Path("/tmp/trace_test.trace.jsonl"), eval_id="e")
        assert isinstance(hook, PreRunStepHook)
        assert isinstance(hook, PostRunStepHook)

    def test_context_variable_lifecycle(self) -> None:
        hook = TracingHook(Path("/tmp/trace_test2.trace.jsonl"), eval_id="e")
        assert get_active_hook() is None
        token = set_active_hook(hook)
        assert get_active_hook() is hook
        reset_active_hook(token)
        assert get_active_hook() is None

    def test_accepts_burr_kwargs(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-test")
        hook.pre_run_step(
            state=MagicMock(),
            action=MagicMock(name="act"),
            app_id="app1",
            partition_key="pk",
            sequence_id=1,
            inputs={},
        )
        hook.post_run_step(
            state=MagicMock(),
            action=MagicMock(name="act"),
            result={},
            sequence_id=1,
            exception=None,
            app_id="app1",
            partition_key="pk",
        )
        lines = trace_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["detail"]["phase"] == "start"
        assert json.loads(lines[1])["detail"]["phase"] == "end"

    def test_with_hooks_accepted_by_builder(self, tmp_path: Path) -> None:
        from burr.core import ApplicationBuilder

        from corallab.judging.graph import build_judge_pipeline_graph

        hook = TracingHook(tmp_path / "t.trace.jsonl", eval_id="e")
        g = build_judge_pipeline_graph().build()
        app = (
            ApplicationBuilder()
            .with_graph(g)
            .with_state(
                run_case=None,
                transcript=None,
                manifest=None,
                evaluation_id="test",
                evaluation_level=1,
                sim_result=None,
                final_result=None,
            )
            .with_hooks(hook)
            .with_entrypoint("prepare_judges")
            .build()
        )
        assert app is not None
