"""Run manifest schema — configures a complete evaluation pipeline run.

A run manifest is the single configuration object that drives one invocation of
``corallab run``, ``corallab sim``, or ``corallab judge``. It selects the models
used at each stage (judge, system-under-test, person response generator, …), declares
the judge fan-out, and configures the simulation and judging models.

Every model role is configured with a :class:`ModelCallConfig` — one reusable
shape carrying a ``model_id`` plus its per-call sampling parameters. Every
sampling field is optional, so a config only carries the params its model
supports: a reasoning model (gpt-5 family, Claude thinking) sets
``reasoning_effort`` / ``verbosity``; a chat model sets ``temperature`` /
``top_p``; neither is forced to carry the other.

Examples:
    Minimal L1 manifest::

        from corallab.schemas.eval.run_manifest import (
            JudgeConfig,
            ModelCallConfig,
            RunManifest,
        )

        manifest = RunManifest(
            run_id="ocd-l1-2026-05-14",
            evaluation_level=1,
            system_under_test=[ModelCallConfig(model_id="azure/gpt-4o-mini")],
            test_case_records_path="test_case_records/ocd_priya_l1.jsonl",
            run_dir="results/ocd-l1-2026-05-14",
            judges=[
                JudgeConfig(
                    judge_type="kl_two_stage",
                    model=ModelCallConfig(model_id="azure/gpt-4o"),
                )
            ],
        )

    ``test_case_records_path`` is resolved by
    :func:`corallab.paths.resolve_data_path`, so write it relative to
    the data repository root — ``test_case_records/ocd_priya_l1.jsonl`` — rather than
    prefixing it with the root's own directory name. ``run_dir`` is the directory
    the run writes into, resolved under the data repository when it is relative.
    The CLI appends ``-YYYYMMDD-HHMMSS`` to that directory name, so
    ``results/ocd-l1-2026-05-14`` lands in
    ``<data-repository>/results/ocd-l1-2026-05-14-<timestamp>``.

    Multiple models under test (sim/run fan out, one transcript set each)::

        manifest = RunManifest(
            ...,
            system_under_test=[
                ModelCallConfig(model_id="azure/gpt-5.1", temperature=1.0,
                                reasoning_effort="high"),
                ModelCallConfig(model_id="anthropic/claude-opus-4-8", temperature=1.0),
            ],
        )

    Judges with options (the Knowledge Library two-stage judge)::

        manifest = RunManifest(
            ...,
            judges=[
                JudgeConfig(
                    judge_type="kl_two_stage",
                    model=ModelCallConfig(model_id="azure/gpt-4o"),
                    options={"group_size": 8},
                ),
            ],
        )

"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ElicitationFidelityConfig",
    "JudgeConfig",
    "ModelCallConfig",
    "RunManifest",
    "SimulationConfig",
    "default_judges",
]

# Default temperatures, applied when a ModelCallConfig leaves temperature unset.
# Not field defaults: the same shape is used for roles with different defaults.
DEFAULT_SUT_TEMPERATURE = 0.7
DEFAULT_SUT_MAX_TOKENS = 4096
DEFAULT_JUDGE_TEMPERATURE = 0.0
# Judge replies include quotes. A low ceiling truncates the JSON.
DEFAULT_JUDGE_MAX_TOKENS = 16000
DEFAULT_PERSON_RESPONSE_MODEL_TEMPERATURE = 0.9
DEFAULT_ELICITATION_FIDELITY_TEMPERATURE = 0.0


def _slug(value: str) -> str:
    """Filesystem-safe slug for a model id (e.g. ``azure/gpt-5.1`` → ``azure-gpt-5-1``)."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class ModelCallConfig(BaseModel):
    """One model plus its per-call sampling parameters.

    Reusable across every model role in a run — the system-under-test, each
    judge pass, and the L3 simulation roles (person response model,
    elicitation fidelity). This is
    *call* config (how a model is invoked); it is distinct from
    :class:`corallab.llm.registry_schema.ModelRoute`, which is *routing* config
    (which endpoint a logical model id resolves to).

    Every sampling field is optional so a config only carries what its model
    supports: a reasoning model (gpt-5 family, Claude thinking) sets
    ``reasoning_effort`` / ``verbosity``; a chat model sets ``temperature`` /
    ``top_p``; neither is forced to carry the other. Unset fields are omitted
    from the request, and calls are made with LiteLLM ``drop_params`` so a
    parameter a given model doesn't support is dropped rather than erroring —
    which lets one run mix reasoning and chat models. ``temperature`` /
    ``max_tokens`` default to ``None`` (omit); call-sites apply a per-role
    default (see ``DEFAULT_*_TEMPERATURE`` constants) when they are unset.
    Unknown keys are an error.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Logical model id (resolved via endpoints.yaml)")
    temperature: float | None = Field(
        default=None,
        description=(
            "Sampling temperature. None omits it; the call-site then applies the "
            "role default. Pin it explicitly for reasoning models that require a "
            "specific value (e.g. 1.0 for the gpt-5 family)."
        ),
    )
    max_tokens: int | None = Field(
        default=None, description="Max tokens for the model response; None = provider default"
    )
    top_p: float | None = Field(
        default=None, description="Nucleus sampling cutoff; omitted from the request when None"
    )
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Reasoning budget for reasoning models "
            "('minimal' | 'low' | 'medium' | 'high'); omitted when None"
        ),
    )
    verbosity: str | None = Field(
        default=None,
        description="Output verbosity for the gpt-5 family ('low' | 'medium' | 'high')",
    )
    label: str | None = Field(
        default=None,
        description=(
            "Optional label disambiguating output files when two system-under-test "
            "entries share a model_id (e.g. same model at two reasoning efforts). "
            "Defaults to a slug of model_id."
        ),
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific completion params forwarded verbatim (e.g. "
            "{'thinking': {'type': 'enabled', 'budget_tokens': 4096}} for Anthropic). "
            "Merged after the typed fields; takes precedence on key collisions."
        ),
    )

    @property
    def output_key(self) -> str:
        """Stable key used to name this model's output file/index entry."""
        return self.label or _slug(self.model_id)

    def temperature_or(self, default: float) -> float:
        """Return the configured temperature, or ``default`` when unset (None)."""
        return self.temperature if self.temperature is not None else default

    def max_tokens_or(self, default: int) -> int:
        """Return the configured max_tokens, or ``default`` when unset (None)."""
        return self.max_tokens if self.max_tokens is not None else default

    def completion_params(self) -> dict[str, Any]:
        """Assemble the optional completion kwargs (top_p, reasoning_effort, …).

        Returns only the parameters that are set, so a model that doesn't
        support one isn't sent a null. ``extra_params`` is merged last and wins
        on collision. ``temperature`` / ``max_tokens`` are passed via their
        dedicated ``LLMClient.complete`` arguments, not here.
        """
        params: dict[str, Any] = {}
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            params["verbosity"] = self.verbosity
        params.update(self.extra_params)
        return params


class JudgeConfig(BaseModel):
    """Configuration for one judge in the evaluation pipeline.

    Each entry is one parallel judge invocation. The manifest's ``judges``
    list determines which judges run and how they are configured.
    Unknown keys are an error.
    """

    model_config = ConfigDict(extra="forbid")

    judge_type: Literal["kl_two_stage"] = Field(
        description="REGISTRY key of the judge to run; 'kl_two_stage' is the only judge"
    )
    model: ModelCallConfig = Field(description="Model this judge calls")
    enabled: bool = Field(default=True, description="Set to false to skip this judge")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Judge-specific options forwarded verbatim to the spec (e.g. the "
            "kl_two_stage judge reads group_size, max_concurrency, pathways_per_call, "
            "prompt_version, pricing); options it does not name are ignored. "
            "Omitted from the file = {}."
        ),
    )


def default_judges(model: ModelCallConfig) -> list[JudgeConfig]:
    """One Knowledge Library two-stage judge, calling ``model``."""
    return [
        JudgeConfig(
            judge_type="kl_two_stage",
            model=model,
        )
    ]


class ElicitationFidelityConfig(BaseModel):
    """Post-conversation check that the simulated person elicited the test case.

    One model call on the finished transcript. There is no per-turn check.
    Unknown keys are an error.
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelCallConfig | None = Field(
        default=None,
        description=(
            "Model for the post-conversation elicitation fidelity check "
            "(default temperature 0.0). When omitted, the check uses "
            "person_response_model."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Run the post-conversation elicitation fidelity check in `corallab run`",
    )
    prompt_version: str | None = Field(
        default=None,
        description=(
            "Prompt file fidelity/prompts/<version>.txt. Omitted uses the checker's current prompt"
        ),
    )


class SimulationConfig(BaseModel):
    """How a run simulates conversations.

    ``batch_size`` is the only concurrency cap. Cases are taken in file order,
    ``batch_size`` at a time. One batch runs to completion — every case in the
    batch is in flight together — and is written to ``run_dir/batch-NNN`` before
    the next batch starts. One field covers L1, L2, and L3.
    Unknown keys are an error.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(
        default=20,
        ge=1,
        description=(
            "How many test cases to simulate at once. Each batch of this size "
            "is written to its own run_dir/batch-NNN directory before the next "
            "batch starts"
        ),
    )
    max_turns: int = Field(default=10, description="Maximum conversation turns for L3 simulation")
    person_response_model: ModelCallConfig | None = Field(
        default=None,
        description=(
            "L3: Model PersonResponseGenerator calls when writing the person's messages "
            "(default temperature 0.9)"
        ),
    )
    elicitation_fidelity: ElicitationFidelityConfig = Field(
        default_factory=ElicitationFidelityConfig,
        description=(
            "Post-conversation check of the simulated person. Omitted means enabled, "
            "and the model falls back to person_response_model"
        ),
    )
    inject_context_into_sut: bool = Field(
        default=False,
        description=(
            "When True, inject person context (persona_description + clinical_context) "
            "into the SUT system prompt before simulation, simulating a clinical handoff "
            "condition."
        ),
    )


class RunManifest(BaseModel):
    """Configuration for a complete evaluation pipeline run.

    Specifies which models to use for judging and simulation, and input/output
    paths. Simulation knobs, including how many cases run at once, live on
    :class:`SimulationConfig`. Unknown keys are an error: a file written for an
    older manifest shape does not load.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Unique identifier for this evaluation run")
    evaluation_level: int = Field(ge=1, le=3, description="Evaluation complexity level (1–3)")

    # --- System Under Test ---
    system_under_test: list[ModelCallConfig] = Field(
        min_length=1,
        description=(
            "Models under evaluation, each with its own sampling params. "
            "`sim` (and `run`) fan out over this list — one transcript set per "
            "entry, written to a per-model output file. A single-entry list "
            "reproduces single-model behavior."
        ),
    )

    # --- I/O ---
    test_case_records_path: str = Field(description="Path to the JSONL file of test case records")
    persona_records_path: str = Field(
        default="persona_records",
        description=(
            "Directory of persona records the test case records' persona_ref values name, one "
            "<persona_id>.json each. Resolved like test_case_records_path"
        ),
    )
    run_dir: str = Field(
        description=(
            "Base directory this run writes into. The CLI appends "
            "-YYYYMMDD-HHMMSS so a later run does not overwrite it. Relative "
            "paths are placed under the data repository. Holds transcripts, "
            "judge results, the log, the Burr database, and copied inputs"
        )
    )

    # --- Simulation ---
    simulation: SimulationConfig = Field(
        default_factory=SimulationConfig,
        description=(
            "Simulation settings. Omitted from the file means the defaults, including batch_size 20"
        ),
    )

    # --- Retry (shared by simulation and judging calls) ---
    retry_max: int = Field(default=2, description="Maximum retry attempts for failed LLM calls")
    retry_backoff_ms: int = Field(default=1000, description="Initial backoff in ms between retries")

    # --- Judges ---
    judges: list[JudgeConfig] = Field(
        min_length=1,
        description="Judges to run, in parallel. Each names the model it calls.",
    )

    # --- Feature flags ---
    debug: bool = Field(default=False, description="Enable debug logging and prompt snapshots")

    @property
    def primary_sut(self) -> ModelCallConfig:
        """The first/active SUT model. Engine call-sites read params from here.

        During fan-out each per-model sub-manifest carries a single-entry
        ``system_under_test``, so the engine always sees one model via
        ``primary_sut``.
        """
        return self.system_under_test[0]

    def per_model_manifests(self) -> list[tuple[str, RunManifest]]:
        """Split into one single-model sub-manifest per SUT entry.

        Returns ``(output_key, manifest)`` pairs in declared order. Each
        sub-manifest is a copy with ``system_under_test`` narrowed to one
        entry, so the existing single-model sim/judge engine runs unchanged.
        """
        return [
            (cfg.output_key, self.model_copy(update={"system_under_test": [cfg]}))
            for cfg in self.system_under_test
        ]

    @model_validator(mode="after")
    def validate_sut_output_keys(self) -> RunManifest:
        keys = [cfg.output_key for cfg in self.system_under_test]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(
                "system_under_test entries produce duplicate output keys "
                f"{dupes}. Give colliding entries distinct `label`s."
            )
        return self

    @model_validator(mode="after")
    def validate_simulation_models(self) -> RunManifest:
        if self.evaluation_level >= 3 and self.simulation.person_response_model is None:
            raise ValueError("simulation.person_response_model is required for L3 evaluation")
        return self
