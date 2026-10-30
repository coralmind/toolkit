"""Tests for RunManifest validation — exercises valid and invalid inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from corallab.paths import eval_run_configs_dir
from corallab.schemas.eval.run_manifest import (
    DEFAULT_SUT_MAX_TOKENS,
    DEFAULT_SUT_TEMPERATURE,
    ModelCallConfig,
    RunManifest,
    default_judges,
)

# ---------------------------------------------------------------------------
# Minimal valid fixture
# ---------------------------------------------------------------------------


def _minimal_run_config_data(level: int = 1) -> dict:
    """Return a minimal valid RunManifest dict for the given level."""
    data = {
        "run_id": "run-test-001",
        "evaluation_level": level,
        "system_under_test": [{"model_id": "test-sut"}],
        "test_case_records_path": "test_case_records/test.jsonl",
        "run_dir": "results/test",
        "judges": [
            {
                "judge_type": "kl_two_stage",
                "model": {"model_id": "test-judge"},
            }
        ],
    }
    if level >= 3:
        data["simulation"] = {"person_response_model": {"model_id": "test-persona"}}
    return data


# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


def test_valid_l1_config():
    config = RunManifest.model_validate(_minimal_run_config_data(level=1))
    assert config.run_id == "run-test-001"
    assert config.evaluation_level == 1


def test_valid_l3_config():
    config = RunManifest.model_validate(_minimal_run_config_data(level=3))
    assert config.evaluation_level == 3
    assert config.simulation.person_response_model is not None
    assert config.simulation.person_response_model.model_id == "test-persona"


def test_defaults_populated():
    """Optional sampling fields are unset (None); call-sites apply role defaults."""
    config = RunManifest.model_validate(_minimal_run_config_data(level=1))
    assert config.primary_sut.temperature is None
    assert config.primary_sut.temperature_or(DEFAULT_SUT_TEMPERATURE) == 0.7
    assert config.simulation.batch_size == 20
    assert config.simulation.max_turns == 10
    assert config.retry_max == 2
    assert config.primary_sut.max_tokens is None
    assert config.primary_sut.max_tokens_or(DEFAULT_SUT_MAX_TOKENS) == 4096
    assert config.debug is False


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------


def test_missing_run_id():
    data = _minimal_run_config_data()
    del data["run_id"]
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "run_id" in str(exc_info.value)


@pytest.mark.parametrize("key", ["judge_model", "seed", "impact", "pathways", "kl", "max_turns"])
def test_retired_run_manifest_keys_are_rejected(key: str) -> None:
    data = _minimal_run_config_data()
    data[key] = "retired"
    with pytest.raises(ValidationError, match=key):
        RunManifest.model_validate(data)


@pytest.mark.parametrize("key", ["depends_on", "prompt_variant", "label"])
def test_retired_judge_keys_are_rejected(key: str) -> None:
    data = _minimal_run_config_data()
    data["judges"] = [
        {
            "judge_type": "kl_two_stage",
            "model": {"model_id": "test-judge"},
            key: "retired",
        }
    ]
    with pytest.raises(ValidationError, match=key):
        RunManifest.model_validate(data)


def test_judge_requires_a_model():
    data = _minimal_run_config_data()
    data["judges"] = [{"judge_type": "kl_two_stage"}]
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "model" in str(exc_info.value)


def test_missing_system_under_test():
    data = _minimal_run_config_data()
    del data["system_under_test"]
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "system_under_test" in str(exc_info.value)


def test_empty_system_under_test_rejected():
    data = _minimal_run_config_data()
    data["system_under_test"] = []
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "system_under_test" in str(exc_info.value)


def test_duplicate_sut_output_keys_rejected():
    data = _minimal_run_config_data()
    data["system_under_test"] = [{"model_id": "azure/gpt-5.1"}, {"model_id": "azure/gpt-5.1"}]
    with pytest.raises(ValidationError, match="duplicate output keys"):
        RunManifest.model_validate(data)


def test_per_model_manifests_splits_by_model():
    data = _minimal_run_config_data()
    data["system_under_test"] = [
        {"model_id": "azure/gpt-5.1", "temperature": 1.0},
        {"model_id": "anthropic/claude-opus-4-8", "label": "opus"},
    ]
    config = RunManifest.model_validate(data)
    per_model = config.per_model_manifests()
    assert [key for key, _ in per_model] == ["azure-gpt-5-1", "opus"]
    assert all(len(sub.system_under_test) == 1 for _, sub in per_model)
    assert per_model[0][1].primary_sut.model_id == "azure/gpt-5.1"


def test_missing_test_case_records_path():
    data = _minimal_run_config_data()
    del data["test_case_records_path"]
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "test_case_records_path" in str(exc_info.value)


def test_missing_run_dir():
    data = _minimal_run_config_data()
    del data["run_dir"]
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "run_dir" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Invalid evaluation_level
# ---------------------------------------------------------------------------


def test_evaluation_level_zero():
    data = _minimal_run_config_data()
    data["evaluation_level"] = 0
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "evaluation_level" in str(exc_info.value)


def test_evaluation_level_five():
    data = _minimal_run_config_data()
    data["evaluation_level"] = 5
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "evaluation_level" in str(exc_info.value)


def test_evaluation_level_wrong_type():
    data = _minimal_run_config_data()
    data["evaluation_level"] = "high"
    with pytest.raises(ValidationError) as exc_info:
        RunManifest.model_validate(data)
    assert "evaluation_level" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Level-specific model requirements
# ---------------------------------------------------------------------------


def test_l3_missing_person_response_model():
    data = _minimal_run_config_data(level=3)
    del data["simulation"]
    with pytest.raises(ValidationError, match="person_response_model"):
        RunManifest.model_validate(data)


def test_l1_does_not_require_sim_models():
    """L1 should not require any simulation models."""
    data = _minimal_run_config_data(level=1)
    assert "simulation" not in data
    config = RunManifest.model_validate(data)
    assert config.simulation.person_response_model is None


# ---------------------------------------------------------------------------
# Wrong types
# ---------------------------------------------------------------------------


def test_system_under_test_temperature_wrong_type():
    data = _minimal_run_config_data()
    data["system_under_test"] = [{"model_id": "test-sut", "temperature": "warm"}]
    with pytest.raises(ValidationError):
        RunManifest.model_validate(data)


def test_batch_size_wrong_type():
    data = _minimal_run_config_data()
    data["simulation"] = {"batch_size": "ten"}
    with pytest.raises(ValidationError):
        RunManifest.model_validate(data)


# ---------------------------------------------------------------------------
# ModelCallConfig
# ---------------------------------------------------------------------------


def test_model_call_completion_params_only_includes_set_fields():
    cfg = ModelCallConfig(model_id="m", reasoning_effort="high")
    params = cfg.completion_params()
    assert params == {"reasoning_effort": "high"}


def test_model_call_completion_params_merges_extra_params_last():
    cfg = ModelCallConfig(
        model_id="m",
        top_p=0.9,
        verbosity="low",
        extra_params={"thinking": {"type": "enabled", "budget_tokens": 4096}},
    )
    params = cfg.completion_params()
    assert params["top_p"] == 0.9
    assert params["verbosity"] == "low"
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_model_call_output_key_defaults_to_model_slug():
    assert ModelCallConfig(model_id="azure/gpt-5.1").output_key == "azure-gpt-5-1"
    assert ModelCallConfig(model_id="azure/gpt-5.1", label="fast").output_key == "fast"


def test_model_call_temperature_and_max_tokens_resolution():
    """Unset sampling fields are None; *_or() applies a caller-supplied default."""
    cfg = ModelCallConfig(model_id="m")
    assert cfg.temperature is None
    assert cfg.max_tokens is None
    assert cfg.temperature_or(0.3) == 0.3
    assert cfg.max_tokens_or(2048) == 2048
    pinned = ModelCallConfig(model_id="m", temperature=1.0, max_tokens=512)
    assert pinned.temperature_or(0.3) == 1.0
    assert pinned.max_tokens_or(2048) == 512


def test_reasoning_and_chat_models_mix_in_one_manifest():
    """A reasoning model carries only reasoning fields; a chat model only chat fields."""
    data = _minimal_run_config_data()
    data["system_under_test"] = [
        {"model_id": "azure/gpt-5.1", "reasoning_effort": "high", "verbosity": "low"},
        {"model_id": "anthropic/claude-opus-4-8", "temperature": 0.7, "top_p": 0.95},
    ]
    config = RunManifest.model_validate(data)
    reasoning, chat = config.system_under_test
    assert reasoning.completion_params() == {"reasoning_effort": "high", "verbosity": "low"}
    assert reasoning.temperature is None
    assert chat.completion_params() == {"top_p": 0.95}
    assert chat.reasoning_effort is None


def test_each_judge_names_its_own_model():
    """Two judges in one run can call different models."""
    data = _minimal_run_config_data()
    data["judges"] = [
        {
            "judge_type": "kl_two_stage",
            "model": {"model_id": "azure/gpt-5.4-mini"},
        },
        {
            "judge_type": "kl_two_stage",
            "model": {"model_id": "azure/gpt-4o", "temperature": 0.2},
        },
    ]
    config = RunManifest.model_validate(data)
    assert config.judges[0].model.model_id == "azure/gpt-5.4-mini"
    assert config.judges[1].model.model_id == "azure/gpt-4o"
    assert config.judges[1].model.temperature == 0.2


def test_retired_judge_is_rejected_at_validation() -> None:
    data = _minimal_run_config_data()
    judge = {
        "judge_type": "kl_two_stage",
        "model": {"model_id": "test-judge"},
    }
    judge["judge_type"] = "context_free"
    data["judges"] = [judge]
    with pytest.raises(ValidationError, match="judge_type"):
        RunManifest.model_validate(data)


def test_default_judges_is_one_kl_two_stage_pass() -> None:
    model = ModelCallConfig(model_id="test-judge")
    [judge] = default_judges(model)
    assert judge.judge_type == "kl_two_stage"
    assert judge.model == model


# ---------------------------------------------------------------------------
# On-disk run configs — guard against schema drift
# ---------------------------------------------------------------------------

_SCHEMA_DIRECTIVE = "# yaml-language-server: $schema=./schema/run_manifest.schema.json\n"
_RUN_CONFIG_FILES = sorted(eval_run_configs_dir().glob("*.yaml"))

# An empty argvalues list makes pytest report one skipped test, not a failure, so
# a moved or renamed directory would take all schema-drift assertions with it and
# nothing would go red.
assert _RUN_CONFIG_FILES, "no committed run configs found — did the manifests move?"


@pytest.mark.parametrize("config_path", _RUN_CONFIG_FILES, ids=lambda p: p.name)
def test_committed_run_config_validates(config_path: Path):
    """Every committed run config validates against the current RunManifest schema.

    Catches stale configs left behind when the schema changes (the inline
    fixtures above never touch the on-disk YAML). The ``schema/`` subdirectory
    holds JSON Schema and is excluded by the top-level ``*.yaml`` glob.
    """
    text = config_path.read_text(encoding="utf-8")
    assert text.startswith(_SCHEMA_DIRECTIVE)
    data = yaml.safe_load(text)
    RunManifest.model_validate(data)


_LAUNCH_MANIFESTS = (
    "l1_local.yaml",
    "l2_local.yaml",
    "l3_local.yaml",
    "l1_azure.yaml",
    "l2_azure.yaml",
    "l3_azure.yaml",
)


def test_launch_manifests_name_the_two_stage_judge_and_the_examples() -> None:
    """Default judges still point at the broken harm-pattern path.

    Pins the six launch files: kl_two_stage, data-root-relative
    test case record paths, and L3 person_response_model. If this fails because a file is
    missing, the glob in test_committed_run_config_validates is not enough
    — that test would skip an empty run_manifests/ directory.
    """
    root = eval_run_configs_dir()
    missing = [name for name in _LAUNCH_MANIFESTS if not (root / name).is_file()]
    assert missing == [], f"launch manifests missing: {missing}"

    for name in _LAUNCH_MANIFESTS:
        data = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        manifest = RunManifest.model_validate(data)
        level = int(name[1])
        backend = "local" if name.endswith("_local.yaml") else "azure"
        assert manifest.evaluation_level == level
        assert (
            manifest.test_case_records_path
            == f"test_case_records/examples/l{level}_launch_smoke.jsonl"
        )
        assert [j.judge_type for j in manifest.judges] == ["kl_two_stage"]
        if backend == "local":
            assert manifest.system_under_test[0].model_id == "local/llama3.2"
            assert manifest.judges[0].model.model_id == "local/llama3.2"
        else:
            assert manifest.system_under_test[0].model_id == "azure/gpt-5.4-mini"
            assert manifest.judges[0].model.model_id == "azure/gpt-5.4-mini"
        if level >= 3:
            assert manifest.simulation.person_response_model is not None
            assert (
                manifest.simulation.person_response_model.model_id
                == manifest.system_under_test[0].model_id
            )
            assert manifest.simulation.max_turns == 4
