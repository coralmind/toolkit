"""Tests for llm/router.py — ModelRouter resolution + LiteLLM Router build."""

from __future__ import annotations

import pytest

from corallab.llm.registry_schema import EndpointConfig, LLMRegistry, ModelRoute
from corallab.llm.router import ModelRouter, UnknownModelError, _retry_policy


def _make_registry(monkeypatch: pytest.MonkeyPatch) -> LLMRegistry:
    monkeypatch.setenv("AZURE_A_KEY", "key-aaa")
    monkeypatch.setenv("AZURE_B_KEY", "key-bbb")
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
                api_base=None,
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


def test_resolve_returns_endpoint_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ModelRouter(_make_registry(monkeypatch))

    judge = router.resolve("judge-on-a")
    assert judge.endpoint_name == "azure-a"
    assert judge.litellm_model == "azure/gpt-5.1"
    assert judge.api_base == "https://a.openai.azure.com"
    assert judge.api_key == "key-aaa"
    assert judge.api_version == "2024-12-01-preview"

    sut = router.resolve("sut-on-b")
    assert sut.endpoint_name == "azure-b"
    assert sut.api_key == "key-bbb"

    persona = router.resolve("persona-on-anthropic")
    assert persona.endpoint_name == "anthropic"
    assert persona.api_base is None
    assert persona.api_key == "key-ant"
    assert persona.api_version is None


def test_resolve_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    router = ModelRouter(_make_registry(monkeypatch))
    with pytest.raises(UnknownModelError, match="not-in-registry"):
        router.resolve("not-in-registry")
    with pytest.raises(UnknownModelError, match=r"azure/gpt-5\.1"):
        router.resolve("azure/gpt-5.1")


def test_empty_registry_resolves_nothing() -> None:
    router = ModelRouter(LLMRegistry())
    with pytest.raises(UnknownModelError):
        router.resolve("anything")


def test_build_router_emits_one_model_list_entry_per_logical_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ModelRouter(_make_registry(monkeypatch))
    # Trigger lazy build.
    built = router._build_router()
    # litellm.Router stores the deployments under .model_list (a list of dicts
    # with 'model_name' and 'litellm_params').
    assert isinstance(built.model_list, list)
    names = sorted(d["model_name"] for d in built.model_list)
    assert names == ["judge-on-a", "persona-on-anthropic", "sut-on-b"]

    # Per-deployment params must carry endpoint credentials, not be looked up
    # globally at call time.
    judge_entry = next(d for d in built.model_list if d["model_name"] == "judge-on-a")
    params = judge_entry["litellm_params"]
    assert params["model"] == "azure/gpt-5.1"
    assert params["api_base"] == "https://a.openai.azure.com"
    assert params["api_key"] == "key-aaa"
    assert params["api_version"] == "2024-12-01-preview"


def test_build_router_forwards_base_model_in_model_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_A_KEY", "key-aaa")
    registry = LLMRegistry(
        endpoints={
            "azure-a": EndpointConfig(
                provider="azure",
                api_base="https://a.openai.azure.com",
                api_key_env="AZURE_A_KEY",
                api_version="2024-12-01-preview",
            ),
        },
        models={
            "judge-on-a": ModelRoute(
                endpoint="azure-a",
                litellm_model="azure/gpt-5.4-nano",
                base_model="azure/gpt-5.4-nano",
            ),
        },
    )
    router = ModelRouter(registry)
    built = router._build_router()
    judge_entry = next(d for d in built.model_list if d["model_name"] == "judge-on-a")
    assert judge_entry["model_info"]["base_model"] == "azure/gpt-5.4-nano"


def test_retry_policy_retries_transient_but_not_caller_side_errors() -> None:
    policy = _retry_policy()
    # Transient failures are retried so a blip does not cost a whole test case.
    assert policy.RateLimitErrorRetries == 3
    assert policy.TimeoutErrorRetries == 2
    assert policy.InternalServerErrorRetries == 2
    # Caller-side / deterministic failures must NOT be retried: re-sending the
    # same prompt just re-trips Azure Content Safety and burns quota.
    assert policy.ContentPolicyViolationErrorRetries == 0
    assert policy.BadRequestErrorRetries == 0
    assert policy.AuthenticationErrorRetries == 0


def test_build_router_applies_retry_policy_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ModelRouter(_make_registry(monkeypatch))
    built = router._build_router()
    # The policy is wired into the LiteLLM Router, and retry_after=0 lets the
    # provider's Retry-After header drive 429 backoff instead of retrying flat.
    assert built.retry_policy is not None
    assert built.retry_policy.ContentPolicyViolationErrorRetries == 0
    assert built.retry_after == 0
