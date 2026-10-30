"""Tests for LLMClient.complete() integration with the endpoint router.

Strategy: install a small in-memory registry via ``set_router`` and then
patch ``ModelRouter.acompletion`` so we can assert which path ran and what
kwargs reached LiteLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

import corallab.llm_client as llm_client_module
from corallab.llm import set_router
from corallab.llm.registry_schema import EndpointConfig, LLMRegistry, ModelRoute
from corallab.llm.router import ModelRouter, UnknownModelError
from corallab.llm_client import llm_client
from corallab.simulation.hooks.tracing import (
    TracingHook,
    reset_active_hook,
    set_active_hook,
)


@dataclass
class _FakeUsage:
    prompt_tokens: int = 7
    completion_tokens: int = 11


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage
    model: str


def _fake_response(text: str = "ok", model: str = "azure/gpt-5.1") -> _FakeResponse:
    return _FakeResponse(
        choices=[_FakeChoice(_FakeMessage(text))],
        usage=_FakeUsage(),
        model=model,
    )


@pytest.fixture(autouse=True)
def _isolate_router() -> None:
    """Each test starts with the global router cleared and ends the same way."""
    set_router(None)
    yield
    set_router(None)


def _three_endpoint_registry(monkeypatch: pytest.MonkeyPatch) -> LLMRegistry:
    monkeypatch.setenv("AZURE_A_KEY", "key-a")
    monkeypatch.setenv("AZURE_B_KEY", "key-b")
    monkeypatch.setenv("ANTHROPIC_KEY", "key-ant")
    return LLMRegistry(
        endpoints={
            "azure-a": EndpointConfig(
                provider="azure",
                api_base="https://a.openai.azure.com",
                api_key_env="AZURE_A_KEY",
                api_version="2024-12-01-preview",
            ),
            "azure-b": EndpointConfig(
                provider="azure",
                api_base="https://b.openai.azure.com",
                api_key_env="AZURE_B_KEY",
                api_version="2024-12-01-preview",
            ),
            "anthropic": EndpointConfig(
                provider="anthropic",
                api_key_env="ANTHROPIC_KEY",
            ),
        },
        models={
            "judge-on-a": ModelRoute(endpoint="azure-a", litellm_model="azure/gpt-5.1"),
            "sut-on-b": ModelRoute(endpoint="azure-b", litellm_model="azure/gpt-5.4-mini"),
            "persona-on-anthropic": ModelRoute(
                endpoint="anthropic", litellm_model="claude-sonnet-4-6"
            ),
        },
    )


async def test_unknown_model_is_rejected() -> None:
    set_router(ModelRouter(LLMRegistry()))

    fake_acompletion = AsyncMock(return_value=_fake_response())
    with (
        patch.object(llm_client_module.litellm, "acompletion", fake_acompletion),
        pytest.raises(UnknownModelError, match=r"azure/gpt-5\.1"),
    ):
        await llm_client.complete(
            model="azure/gpt-5.1",
            messages=[{"role": "user", "content": "hi"}],
            trace_role="judge",
        )

    fake_acompletion.assert_not_awaited()


# ---------------------------------------------------------------------------
# Registry path: per-call api_base / api_key / api_version reach LiteLLM
# ---------------------------------------------------------------------------


async def test_registered_model_routes_through_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ModelRouter(_three_endpoint_registry(monkeypatch))
    set_router(router)

    fake_router_acompletion = AsyncMock(return_value=_fake_response(model="azure/gpt-5.4-mini"))
    fake_litellm_acompletion = AsyncMock(return_value=_fake_response())

    with (
        patch.object(router, "acompletion", fake_router_acompletion),
        patch.object(llm_client_module.litellm, "acompletion", fake_litellm_acompletion),
    ):
        await llm_client.complete(
            model="sut-on-b",
            messages=[{"role": "user", "content": "hi"}],
            trace_role="sut",
        )

    fake_router_acompletion.assert_awaited_once()
    fake_litellm_acompletion.assert_not_awaited()
    # LiteLLM Router resolves the per-deployment api_base/api_key itself
    # from the model_list entry — the LLMClient passes the *logical* id so
    # Router can look up the right deployment.
    kwargs = fake_router_acompletion.await_args.kwargs
    assert kwargs["model"] == "sut-on-b"


async def test_three_endpoint_run_routes_each_call_to_correct_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Acceptance criterion: a single 'run' can mix three endpoints."""
    router = ModelRouter(_three_endpoint_registry(monkeypatch))
    set_router(router)

    seen: list[str] = []

    async def _record(**kwargs):
        seen.append(kwargs["model"])
        return _fake_response()

    hook = TracingHook(tmp_path / "trace.jsonl", eval_id="eval-123")
    token = set_active_hook(hook)
    try:
        with patch.object(router, "acompletion", side_effect=_record):
            for logical in ["judge-on-a", "sut-on-b", "persona-on-anthropic"]:
                await llm_client.complete(
                    model=logical,
                    messages=[{"role": "user", "content": "hi"}],
                    trace_role=logical,
                )
    finally:
        reset_active_hook(token)

    assert seen == ["judge-on-a", "sut-on-b", "persona-on-anthropic"]

    # Trace records the resolving endpoint for each call.
    lines = [line for line in (tmp_path / "trace.jsonl").read_text().splitlines() if line.strip()]
    import json

    llm_entries = [json.loads(line) for line in lines if json.loads(line)["type"] == "llm_call"]
    assert len(llm_entries) == 3
    by_role = {e["role"]: e for e in llm_entries}
    assert by_role["judge-on-a"]["endpoint_name"] == "azure-a"
    assert by_role["judge-on-a"]["resolved_litellm_model"] == "azure/gpt-5.1"
    assert by_role["sut-on-b"]["endpoint_name"] == "azure-b"
    assert by_role["sut-on-b"]["resolved_litellm_model"] == "azure/gpt-5.4-mini"
    assert by_role["persona-on-anthropic"]["endpoint_name"] == "anthropic"
    assert by_role["persona-on-anthropic"]["resolved_litellm_model"] == "claude-sonnet-4-6"


async def test_unknown_model_writes_no_trace(tmp_path) -> None:
    set_router(ModelRouter(LLMRegistry()))

    hook = TracingHook(tmp_path / "trace.jsonl", eval_id="eval-unknown")
    token = set_active_hook(hook)
    try:
        with pytest.raises(UnknownModelError):
            await llm_client.complete(
                model="azure/gpt-5.1",
                messages=[{"role": "user", "content": "hi"}],
                trace_role="judge",
            )
    finally:
        reset_active_hook(token)

    assert not (tmp_path / "trace.jsonl").exists()
