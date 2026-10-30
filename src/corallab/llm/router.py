"""ModelRouter — thin wrapper around ``litellm.Router``.

Why wrap LiteLLM's Router instead of using it directly?

1. We want ``LLMClient.complete()`` to know which **endpoint** served a
   call so the trace (``*.trace.jsonl``) records ``endpoint_name`` and
   ``resolved_litellm_model``. ``ResolvedModel`` is the small object that
   carries that information across the call boundary.

2. A model id that is not declared in the registry raises
   ``UnknownModelError``. Calls do not fall through to process env vars.

3. We want one process-wide singleton router constructed lazily so
   tests can swap it via ``set_router(...)`` without touching imports.

Wrapping Router (not replacing it) means we still get its retries,
weighted routing, cooldowns, and fallbacks for free if we add them to
the registry later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from corallab import paths
from corallab.llm.registry_loader import load_registry, resolve_api_key
from corallab.llm.registry_schema import LLMRegistry

if TYPE_CHECKING:
    from litellm import Router


def _retry_policy() -> Any:
    """Per-error-type retry counts for the LiteLLM Router.

    Retries only *transient* failures. A failed SUT call drops the whole
    test case from the results file (see ``sut_runner``), so transient
    blips — rate limits, timeouts, upstream 5xx — should not cost a case.

    **Content-policy violations are NOT retried** (count 0): Azure Content
    Safety rejects a given prompt deterministically, so re-sending the same
    messages just re-trips the filter and burns quota. The same applies to
    other ``BadRequestError`` / auth failures — they are caller-side and a
    retry cannot fix them.
    """
    from litellm.router import RetryPolicy

    return RetryPolicy(
        RateLimitErrorRetries=3,
        TimeoutErrorRetries=2,
        InternalServerErrorRetries=2,
        ContentPolicyViolationErrorRetries=0,
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
    )


class UnknownModelError(LookupError):
    """A model id is not declared in the endpoint registry."""

    def __init__(self, model_id: str, *, detail: str | None = None) -> None:
        self.model_id = model_id
        message = detail or (
            f"Model {model_id!r} is not in the endpoint registry. "
            "Declare it under models in endpoints.yaml."
        )
        super().__init__(message)


@dataclass(frozen=True)
class ResolvedModel:
    """Result of resolving a logical model id through the registry.

    The four endpoint-credential fields are passed straight to
    ``litellm.Router.acompletion`` as kwargs. ``endpoint_name`` and
    ``litellm_model`` are stamped on the trace so debugging "why did this
    cost X / hit a 429" is possible after the fact.
    """

    logical_id: str
    endpoint_name: str
    litellm_model: str
    api_base: str | None
    api_key: str | None
    api_version: str | None


class ModelRouter:
    """Resolves logical model ids and forwards calls via ``litellm.Router``.

    Construct from an :class:`LLMRegistry`. Resolution is pure-Python
    (dict lookup); the underlying LiteLLM ``Router`` is built lazily on
    first ``acompletion`` call so importing the registry never triggers
    LiteLLM's heavy initialization.
    """

    def __init__(self, registry: LLMRegistry) -> None:
        self._registry = registry
        self._router: Router | None = None

    # -- Resolution --------------------------------------------------------

    def resolve(self, logical_id: str) -> ResolvedModel:
        """Resolve a logical model id to its endpoint and credentials.

        Raises :class:`UnknownModelError` when the id is not in the registry.
        """
        model = self._registry.models.get(logical_id)
        if model is None:
            raise UnknownModelError(logical_id)
        endpoint = self._registry.endpoints.get(model.endpoint)
        if endpoint is None:
            raise UnknownModelError(
                logical_id,
                detail=(f"Model {logical_id!r} references unknown endpoint {model.endpoint!r}."),
            )
        return ResolvedModel(
            logical_id=logical_id,
            endpoint_name=model.endpoint,
            litellm_model=model.litellm_model,
            api_base=endpoint.api_base,
            api_key=resolve_api_key(endpoint),
            api_version=endpoint.api_version,
        )

    # -- LiteLLM Router (lazy) --------------------------------------------

    def _build_router(self) -> Router:
        """Build the LiteLLM Router model_list once, on first use."""
        from litellm import Router as LiteLLMRouter

        model_list: list[dict[str, Any]] = []
        for logical_id, model in self._registry.models.items():
            endpoint = self._registry.endpoints[model.endpoint]
            params: dict[str, Any] = {"model": model.litellm_model}
            if endpoint.api_base is not None:
                params["api_base"] = endpoint.api_base
            api_key = resolve_api_key(endpoint)
            if api_key is not None:
                params["api_key"] = api_key
            if endpoint.api_version is not None:
                params["api_version"] = endpoint.api_version
            entry: dict[str, Any] = {"model_name": logical_id, "litellm_params": params}
            if model.base_model is not None:
                entry["model_info"] = {"base_model": model.base_model}
            model_list.append(entry)

        return LiteLLMRouter(
            model_list=model_list,
            retry_policy=_retry_policy(),
            # Honor the provider's `Retry-After` header on 429s rather than
            # retrying immediately. 0 means "use the header if present, else
            # exponential backoff" — which is what we want for Azure rate limits.
            retry_after=0,
        )

    async def acompletion(self, **kwargs: Any) -> Any:
        """Proxy to ``litellm.Router.acompletion`` (lazy-builds the router)."""
        if self._router is None:
            self._router = self._build_router()
        return await self._router.acompletion(**kwargs)

    @property
    def registry(self) -> LLMRegistry:
        return self._registry


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_router_singleton: ModelRouter | None = None


def get_router() -> ModelRouter:
    """Return the process-wide router, lazily loading the default registry.

    Loads ``paths.endpoints_registry_path()``
    (``<data-repository>/endpoints/endpoints.yaml``). A missing file raises
    ``FileNotFoundError``.
    """
    global _router_singleton
    if _router_singleton is None:
        registry = load_registry(paths.endpoints_registry_path())
        _router_singleton = ModelRouter(registry)
    return _router_singleton


def set_router(router: ModelRouter | None) -> None:
    """Override the process-wide router. Pass ``None`` to reset."""
    global _router_singleton
    _router_singleton = router
