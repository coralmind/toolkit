"""Pydantic schemas for the LLM endpoint + model registry.

The registry YAML (``data-repository/endpoints/endpoints.yaml``) declares:

  endpoints:
    azure-eastus:
      provider: azure
      api_base: ${AZURE_EASTUS_API_BASE}
      api_key_env: AZURE_EASTUS_API_KEY
      api_version: "2024-12-01-preview"

  models:
    judge-gpt-51:
      endpoint: azure-eastus
      litellm_model: azure/gpt-5.1

The schema is intentionally narrow: only env-var **names** (``api_key_env``)
and ``${VAR}`` interpolation strings appear in the YAML so secrets stay in
the environment. ``registry_loader.load_registry()`` does the env-var lookup
+ interpolation when the file is read.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EndpointConfig(BaseModel):
    """One LLM access point (provider + creds + base URL)."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        description=(
            "LiteLLM provider family. Common values: 'azure', 'openai', "
            "'anthropic', 'bedrock'. Used for documentation and validation only; "
            "the actual provider is selected by the prefix on `litellm_model`."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL for the endpoint. Supports ${ENV_VAR} interpolation. "
            "May be None for providers that don't use a base URL (e.g. OpenAI direct)."
        ),
    )
    api_key_env: str | None = Field(
        default=None,
        description=(
            "Name of the environment variable holding the API key. The key "
            "itself never appears in the YAML. May be None for endpoints that "
            "require no auth (e.g. local Ollama)."
        ),
    )
    api_version: str | None = Field(
        default=None,
        description="API version (Azure-specific, e.g. '2024-12-01-preview').",
    )


class ModelRoute(BaseModel):
    """A logical model id pointing at one endpoint + a LiteLLM model string.

    This is endpoint-routing config (*where* a model lives), distinct from
    ``corallab.schemas.eval.run_manifest.ModelCallConfig`` (*how* a model is
    called: sampling params).
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(
        description="Name of the endpoint (key in `endpoints:`) this model lives on.",
    )
    litellm_model: str = Field(
        description=(
            "The actual model string LiteLLM expects, e.g. 'azure/gpt-5.1', "
            "'claude-sonnet-4-6', 'openai/llama3:70b'."
        ),
    )
    base_model: str | None = Field(
        default=None,
        description=(
            "Optional LiteLLM catalog id for Azure custom deployment names. "
            "Forwarded as model_info.base_model so LiteLLM can resolve max "
            "tokens, capabilities, and cost tracking without logging "
            "'Could not identify azure model' warnings."
        ),
    )


class LLMRegistry(BaseModel):
    """Top-level registry: named endpoints + named models."""

    model_config = ConfigDict(extra="forbid")

    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    models: dict[str, ModelRoute] = Field(default_factory=dict)
