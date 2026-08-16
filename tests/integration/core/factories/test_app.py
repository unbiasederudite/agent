import json
from pathlib import Path

import pytest

from agent.core.exceptions import ConfigError
from agent.core.factories.app import build_llm_registry


def test_build_given_valid_config_registers_llm(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    registry = build_llm_registry(config_path)

    assert registry.get("openai/gpt-4o") is not None


def test_build_given_duplicate_model_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps({"llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o"}]})
    )

    with pytest.raises(ConfigError):
        build_llm_registry(config_path)


def test_build_given_missing_file_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "does_not_exist.json"

    with pytest.raises(ConfigError):
        build_llm_registry(config_path)


def test_build_given_invalid_json_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text("not json")

    with pytest.raises(ConfigError):
        build_llm_registry(config_path)
