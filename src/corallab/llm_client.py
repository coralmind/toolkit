"""LLM calls through LiteLLM.

A model id must be declared in ``data-repository/endpoints/endpoints.yaml``.
``corallab.llm.ModelRouter``
resolves it to an endpoint and supplies ``api_base``, ``api_key``, and
``api_version`` from that entry. An id that is not declared raises
``corallab.llm.UnknownModelError``.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from corallab.simulation.common.dry_run import DryRunCollector

# Dry-run collector. Set with set_dry_run_context() around asyncio.run().

_dry_run_ctx: contextvars.ContextVar[DryRunCollector | None] = contextvars.ContextVar(
    "_dry_run_ctx", default=None
)


def set_dry_run_context(ctx: DryRunCollector | None) -> contextvars.Token:
    """Activate dry-run mode. Returns a token; pass to reset_dry_run_context() in finally."""
    return _dry_run_ctx.set(ctx)


def reset_dry_run_context(token: contextvars.Token) -> None:
    """Restore the prior dry-run context using the token from set_dry_run_context()."""
    _dry_run_ctx.reset(token)


def get_dry_run_context() -> DryRunCollector | None:
    """Return the active DryRunCollector, or None if not in dry-run mode."""
    return _dry_run_ctx.get()


# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# CORALMIND_GEMINI_DISABLE_SAFETY=1 sends BLOCK_NONE on Gemini calls.
# Gemini's default filter can block a clinical prompt (finish_reason=content_filter).
# Passed as a per-call safety_settings kwarg: litellm's module global is skipped
# on the Router path. Off by default, so a Gemini system under test keeps its filter.
_GEMINI_DISABLE_SAFETY = os.environ.get("CORALMIND_GEMINI_DISABLE_SAFETY") == "1"
_GEMINI_SAFETY_BLOCK_NONE = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
]

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    text: str
    model: str
    usage: dict[str, int] | None = None  # {"prompt_tokens": ..., "completion_tokens": ...}


class TokenAccumulator:
    """Accumulates real token usage across a run for post-run summary reporting.

    Tracks totals and per-(role, model) breakdowns.  Activated via
    set_token_accumulator() before asyncio.run() and read after completion.
    """

    def __init__(self) -> None:
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cache: int = 0
        # {(role, model): [input, output, cache]}
        self._by_role_model: dict[tuple[str, str], list[int]] = {}

    def record(
        self,
        *,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_tokens: int,
    ) -> None:
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cache += cache_tokens
        key = (role, model)
        if key not in self._by_role_model:
            self._by_role_model[key] = [0, 0, 0]
        self._by_role_model[key][0] += input_tokens
        self._by_role_model[key][1] += output_tokens
        self._by_role_model[key][2] += cache_tokens

    @property
    def total_input(self) -> int:
        return self._total_input

    @property
    def total_output(self) -> int:
        return self._total_output

    @property
    def total_cache(self) -> int:
        return self._total_cache

    @property
    def breakdown(self) -> dict[tuple[str, str], tuple[int, int, int]]:
        """Return {(role, model): (input, output, cache)} sorted by role then model."""
        return {k: (v[0], v[1], v[2]) for k, v in sorted(self._by_role_model.items())}

    def merge(self, other: TokenAccumulator) -> None:
        """Add ``other``'s totals into this accumulator."""
        self._total_input += other._total_input
        self._total_output += other._total_output
        self._total_cache += other._total_cache
        for key, counts in other._by_role_model.items():
            slot = self._by_role_model.setdefault(key, [0, 0, 0])
            slot[0] += counts[0]
            slot[1] += counts[1]
            slot[2] += counts[2]

    def summary_lines(self, label: str) -> list[str]:
        """Render the token-usage summary as plain text lines.

        Pure formatting — no logging, no console. Callers (e.g. the CLI) decide
        how to present the result, keeping this module free of presentation deps.
        """
        col_role = max((len(r) for r, _ in self._by_role_model), default=4)
        col_role = max(col_role, len("Role"), len("TOTAL"))
        col_model = max((len(m) for _, m in self._by_role_model), default=5)
        col_model = max(col_model, len("Model"))

        header = (
            f"{'Role':<{col_role}}  {'Model':<{col_model}}"
            f"  {'Input':>12}  {'Output':>10}  {'Cache read':>12}"
        )
        sep = "─" * len(header)

        out = [
            f"Token usage — {label}",
            sep,
            header,
            sep,
        ]
        for (role, model), (inp, outp, cache) in self.breakdown.items():
            out.append(
                f"{role:<{col_role}}  {model:<{col_model}}  {inp:>12,}  {outp:>10,}  {cache:>12,}"
            )
        out.append(sep)
        out.append(
            f"{'TOTAL':<{col_role}}  {''.ljust(col_model)}"
            f"  {self._total_input:>12,}  {self._total_output:>10,}  {self._total_cache:>12,}"
        )
        return out


# Token tally for the current run. Set around asyncio.run() in the CLI runners.

_token_accumulator_ctx: contextvars.ContextVar[TokenAccumulator | None] = contextvars.ContextVar(
    "_token_accumulator_ctx", default=None
)


def set_token_accumulator(acc: TokenAccumulator | None) -> contextvars.Token:
    """Activate a TokenAccumulator for the current context. Returns a reset token."""
    return _token_accumulator_ctx.set(acc)


def reset_token_accumulator(token: contextvars.Token) -> None:
    """Restore the prior accumulator context."""
    _token_accumulator_ctx.reset(token)


def get_token_accumulator() -> TokenAccumulator | None:
    """Return the active TokenAccumulator, or None if not set."""
    return _token_accumulator_ctx.get()


class LLMClient:
    """Async LLM client backed by LiteLLM.

    All LLM calls in the system flow through this class.
    """

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = 0.0,
        trace_role: str,
        trace_turn: int | None = None,
        trace_test_case_id: str | None = None,
        prompt_cache_key: str | None = None,
        cache_prefix: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call an LLM model via LiteLLM.

        Args:
            model: Model identifier (e.g. "claude-sonnet-4-6", "azure/gpt-4o").
            messages: List of {"role": "user"|"assistant", "content": "..."}.
            system: Optional system prompt (prepended as a system message).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature. Pass None to omit (model default).
            trace_role: Caller identity for tracing (e.g. "sut", "person_response_generator",
                "elicitation_fidelity", "coralmind_s1_<group_id>", "coralmind_s2_<ids>").
            trace_turn: Conversation turn this call is associated with (None for judge calls).
            trace_test_case_id: Originating test case id when the call belongs to a
                per-test-case flow. Recorded on the trace row so consumers can split
                rows per test case in multi-case batches. Pass ``None`` for run-level
                calls that aren't attributable to one case.
            prompt_cache_key: Optional OpenAI cache routing key. Calls sharing the same
                key route to the same cache shard, improving hit rates for >15 RPM
                workloads with shared prefixes. Ignored by non-OpenAI providers.
            extra_params: Optional provider completion params (e.g. ``top_p``,
                ``reasoning_effort``, ``verbosity``, or provider-specific keys
                like Anthropic ``thinking``). Forwarded to LiteLLM with
                ``drop_params=True`` so a param the resolved model doesn't
                support is dropped rather than raising.

        Returns:
            LLMResponse with the text content.

        Raises:
            litellm.RateLimitError: Rate limited by provider.
            litellm.APIConnectionError: Network/connection failure.
            litellm.APIError: Other API errors.
        """
        litellm_messages: list[dict[str, Any]] = []
        if system:
            litellm_messages.append({"role": "system", "content": system})
        litellm_messages.extend(messages)

        logger.debug(
            "LLM REQUEST: model=%s max_tokens=%s temperature=%.2f messages=%d system=%s",
            model,
            max_tokens or "model_default",
            temperature,
            len(messages),
            "yes" if system else "no",
        )

        from corallab.llm import get_router

        resolved = get_router().resolve(model)
        # Feature gates use the litellm model string, not the logical id.
        provider_model = resolved.litellm_model

        # Claude: split the last user message and mark cache_prefix with cache_control.
        # The prefix must be a leading substring; otherwise the call is sent unchanged.
        if cache_prefix and "claude" in provider_model.lower() and litellm_messages:
            last = litellm_messages[-1]
            content = last.get("content")
            if (
                last.get("role") == "user"
                and isinstance(content, str)
                and content.startswith(cache_prefix)
                and len(cache_prefix) < len(content)
            ):
                # New dict: extend() shallow-copies, so mutating last would edit the
                # caller's message.
                litellm_messages[-1] = {
                    **last,
                    "content": [
                        {
                            "type": "text",
                            "text": cache_prefix,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": content[len(cache_prefix) :]},
                    ],
                }
            else:
                logger.debug(
                    "llm_client: cache_prefix not applied (model=%s, prefix-matches=%s) — "
                    "sending plain content",
                    provider_model,
                    isinstance(content, str) and content.startswith(cache_prefix),
                )

        kwargs: dict[str, Any] = {
            "model": provider_model,
            "messages": litellm_messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            if provider_model.startswith("azure/"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

        # OpenAI and Azure accept prompt_cache_key. Anthropic returns 400 on the extra field.
        if prompt_cache_key is not None:
            if provider_model.startswith(("azure/", "openai/", "gpt-")):
                kwargs["extra_body"] = {"prompt_cache_key": prompt_cache_key}
            else:
                logger.debug(
                    "llm_client: dropping prompt_cache_key for non-OpenAI model %s",
                    provider_model,
                )

        # drop_params: an unsupported param is dropped instead of raising.
        if extra_params:
            kwargs.update({k: v for k, v in extra_params.items() if v is not None})
            kwargs["drop_params"] = True

        # Per-call: the Router does not see LiteLLM's module-global safety settings.
        if _GEMINI_DISABLE_SAFETY and provider_model.startswith("gemini/"):
            kwargs.setdefault("safety_settings", _GEMINI_SAFETY_BLOCK_NONE)

        # Dry run records the prompt and returns without calling the provider.
        dry_run_collector = get_dry_run_context()
        if dry_run_collector is not None:
            from corallab.simulation.hooks import get_active_test_case_id

            prompt_tokens = litellm.token_counter(model=provider_model, messages=litellm_messages)
            resolved_tc_id = trace_test_case_id or get_active_test_case_id()
            stub_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "cached_tokens": 0,
            }
            self._trace_call(
                model=provider_model,
                role=trace_role,
                turn=trace_turn,
                test_case_id=resolved_tc_id,
                system=system,
                # Full messages list for debug (normal path uses last message only).
                prompt=str(litellm_messages),
                response_text="",
                latency_ms=0,
                usage=stub_usage,
                cached_tokens=0,
                endpoint_name=resolved.endpoint_name,
                resolved_litellm_model=resolved.litellm_model,
            )
            dry_run_collector.record(
                cache_key=prompt_cache_key,
                trace_role=trace_role,
                prompt_tokens=prompt_tokens,
                provider_model=provider_model,
            )
            return LLMResponse(text="", model=provider_model, usage=stub_usage)

        t0 = time.monotonic()
        # The logical id is what Router looks up in model_list. It then
        # injects that entry's api_base, api_key, and api_version.
        kwargs["model"] = resolved.logical_id
        response = await get_router().acompletion(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Extract raw response details
        choice = response.choices[0] if response.choices else None
        finish_reason = choice.finish_reason if choice else "no_choice"
        text = choice.message.content or "" if choice else ""
        usage = None
        cached_tokens: int | None = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
            # OpenAI: usage.prompt_tokens_details.cached_tokens
            # Anthropic: usage.cache_read_input_tokens
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", None)
            if cached_tokens is None:
                cached_tokens = getattr(response.usage, "cache_read_input_tokens", None)
            if cached_tokens is not None:
                usage["cached_tokens"] = cached_tokens

        # Log full response trace
        hidden = getattr(response, "_hidden_params", {}) or {}
        api_base = hidden.get("api_base", "unknown")
        http_status = getattr(response, "_response_status", None)

        logger.info(
            "LLM RESPONSE: model=%s finish_reason=%s tokens=%s http_status=%s api_base=%s",
            response.model or model,
            finish_reason,
            usage,
            http_status or "n/a",
            api_base,
        )

        if finish_reason == "content_filter":
            logger.warning(
                "LLM CONTENT FILTERED: model=%s — response blocked by provider content filter. "
                "Check Azure Content Safety settings for this deployment.",
                model,
            )

        if not text:
            logger.warning(
                "LLM EMPTY RESPONSE: model=%s finish_reason=%s — "
                "response content is empty (possible content filter or refusal)",
                model,
                finish_reason,
            )

        logger.debug(
            "LLM RESPONSE BODY (first 500 chars): %s",
            text[:500] if text else "<empty>",
        )

        # Explicit arg wins over the ambient ContextVar set by the Burr hook.
        # Judge callers thread the id explicitly; simulation actions inherit it
        # from state via TracingHook.pre_run_step → _active_test_case_id.
        if trace_test_case_id is None:
            from corallab.simulation.hooks import get_active_test_case_id

            trace_test_case_id = get_active_test_case_id()

        self._trace_call(
            model=response.model or model,
            role=trace_role,
            turn=trace_turn,
            test_case_id=trace_test_case_id,
            system=system,
            prompt=litellm_messages[-1].get("content", "") if litellm_messages else "",
            response_text=text,
            latency_ms=latency_ms,
            usage=usage,
            cached_tokens=cached_tokens,
            endpoint_name=resolved.endpoint_name,
            resolved_litellm_model=resolved.litellm_model,
        )

        acc = get_token_accumulator()
        if acc is not None and usage is not None:
            acc.record(
                role=trace_role,
                model=model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                cache_tokens=usage.get("cached_tokens", 0),
            )

        return LLMResponse(
            text=text,
            model=response.model or model,
            usage=usage,
        )

    @staticmethod
    def _trace_call(
        *,
        model: str,
        role: str,
        turn: int | None,
        test_case_id: str | None,
        system: str | None,
        prompt: str,
        response_text: str,
        latency_ms: int,
        usage: dict[str, int] | None,
        cached_tokens: int | None = None,
        endpoint_name: str | None = None,
        resolved_litellm_model: str | None = None,
    ) -> None:
        """Write an LLM call to the active trace hook, if one is set."""
        from corallab.simulation.hooks import get_active_hook

        hook = get_active_hook()
        if hook is None:
            return
        hook.log_llm_call(
            role=role,
            turn=turn,
            test_case_id=test_case_id,
            model=model,
            system=system,
            prompt=prompt,
            response=response_text,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens") if usage else None,
            completion_tokens=usage.get("completion_tokens") if usage else None,
            cached_tokens=cached_tokens,
            endpoint_name=endpoint_name,
            resolved_litellm_model=resolved_litellm_model,
        )


# Module-level singleton — import and use directly
llm_client = LLMClient()
