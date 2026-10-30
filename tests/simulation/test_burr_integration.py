"""Tests for Burr integration components."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from corallab.simulation.hooks import TracingHook


class TestTracingHook:
    """Tests for the TracingHook lifecycle hook."""

    def test_log_event_writes_jsonl(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-abc123")

        hook.pre_run_step(
            state=MagicMock(),
            action=MagicMock(name="test_action"),
        )

        assert trace_path.exists()
        lines = trace_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "event"
        assert entry["eval_id"] == "eval-abc123"
        assert entry["seq"] == 1

    def test_log_llm_call_writes_correct_format(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-abc123")

        hook.log_llm_call(
            role="sut",
            turn=1,
            model="gpt-4",
            prompt="Hello",
            response="Hi there",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=20,
        )

        lines = trace_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["type"] == "llm_call"
        assert entry["model"] == "gpt-4"
        assert entry["latency_ms"] == 100

    def test_seq_is_monotonically_increasing(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-abc123")

        for _ in range(5):
            hook.pre_run_step(state=MagicMock(), action=MagicMock(name="a"))

        lines = trace_path.read_text().strip().splitlines()
        seqs = [json.loads(line)["seq"] for line in lines]
        assert seqs == [1, 2, 3, 4, 5]

    def test_eval_id_setter(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="pending")
        hook.eval_id = "eval-final"

        hook.pre_run_step(state=MagicMock(), action=MagicMock(name="a"))

        entry = json.loads(trace_path.read_text().strip())
        assert entry["eval_id"] == "eval-final"

    def test_log_llm_call_records_test_case_id(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-abc")
        hook.log_llm_call(
            role="sut",
            model="gpt-4",
            prompt="p",
            response="r",
            latency_ms=1,
            test_case_id="aimhef-abcd1234",
        )
        entry = json.loads(trace_path.read_text().strip())
        assert entry["test_case_id"] == "aimhef-abcd1234"

    def test_pre_run_step_sets_active_test_case_id_from_state(self, tmp_path: Path) -> None:
        """Burr hook reads state['test_case_id'] and sets the ambient ContextVar
        so LLM calls inside the action pick it up without signature plumbing."""
        import contextvars

        from corallab.simulation.hooks import get_active_test_case_id

        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-abc")

        state = MagicMock()
        state.get = MagicMock(return_value="aimhef-from-state")

        ctx = contextvars.copy_context()

        def _run_in_ctx() -> str | None:
            hook.pre_run_step(state=state, action=MagicMock(name="a"))
            return get_active_test_case_id()

        assert ctx.run(_run_in_ctx) == "aimhef-from-state"

    def test_pre_run_step_leaves_ambient_unset_when_state_has_no_tc_id(
        self, tmp_path: Path
    ) -> None:
        """Run-level actions whose state lacks test_case_id must not clobber
        a previously set ambient id."""
        import contextvars

        from corallab.simulation.hooks import (
            get_active_test_case_id,
            set_active_test_case_id,
        )

        trace_path = tmp_path / "test.trace.jsonl"
        hook = TracingHook(trace_path, eval_id="eval-abc")

        state = MagicMock()
        state.get = MagicMock(return_value=None)

        ctx = contextvars.copy_context()

        def _run_in_ctx() -> str | None:
            set_active_test_case_id("aimhef-prev")
            hook.pre_run_step(state=state, action=MagicMock(name="a"))
            return get_active_test_case_id()

        assert ctx.run(_run_in_ctx) == "aimhef-prev"


class TestGraphBuilders:
    """Tests that graph builders produce valid Burr graphs."""

    def test_judge_graph_builds(self) -> None:
        from corallab.judging.graph import build_judge_pipeline_graph

        graph = build_judge_pipeline_graph().build()
        action_names = [a.name for a in graph.actions]
        assert "prepare_judges" in action_names
        assert "fan_out_judges" in action_names
        assert "merge_judges" in action_names
        assert "assemble" not in action_names
        assert "system_under_test" not in action_names
        assert "simulate" not in action_names

    def test_sim_batch_graph_builds(self) -> None:
        from corallab.simulation.batch.graph import build_sim_batch_graph
        from corallab.simulation.graph import build_sim_subgraph

        per_case = build_sim_subgraph(1)
        graph = build_sim_batch_graph(per_case).build()
        action_names = [a.name for a in graph.actions]
        assert "load" not in action_names
        assert "simulate" in action_names
        assert "write" in action_names

    def test_judge_batch_graph_builds(self) -> None:
        from corallab.judging.graph import build_judge_subgraph
        from corallab.simulation.batch.graph import build_eval_batch_graph

        per_case = build_judge_subgraph()
        graph = build_eval_batch_graph(per_case).build()
        action_names = [a.name for a in graph.actions]
        assert "load" in action_names
        assert "judge" in action_names
        assert "write" in action_names

    def test_per_case_sim_graph_l1(self) -> None:
        from corallab.simulation.graph import build_sim_subgraph

        rg = build_sim_subgraph(1)
        assert rg.entrypoint == "system_under_test"
        assert rg.halt_after == ["system_under_test"]

    def test_per_case_sim_graph_l3(self) -> None:
        from corallab.simulation.graph import build_sim_subgraph

        rg = build_sim_subgraph(3)
        assert rg.entrypoint == "simulate"
        assert rg.halt_after == ["simulate"]

    def test_per_case_judge_graph(self) -> None:
        from corallab.judging.graph import build_judge_subgraph

        rg = build_judge_subgraph()
        assert rg.entrypoint == "prepare_judges"
        assert rg.halt_after == ["merge_judges"]
