import pytest
from pydantic import ValidationError

from agent.core.models.config import AppConfig, LLMConfig, LoggingConfig


def test_llm_config_given_model_string_constructs():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.model == "openai/gpt-4o"


def test_logging_config_given_no_args_defaults_to_info():
    config = LoggingConfig()

    assert config.level == "INFO"


def test_logging_config_given_level_constructs():
    config = LoggingConfig(level="DEBUG")

    assert config.level == "DEBUG"


def test_logging_config_given_invalid_level_raises_validation_error():
    with pytest.raises(ValidationError):
        LoggingConfig(level="verbose")


def test_app_config_given_valid_json_constructs():
    raw = '{"llms": [{"model": "openai/gpt-4o"}], "logging": {"level": "DEBUG"}}'

    config = AppConfig.model_validate_json(raw)

    assert config.llms[0].model == "openai/gpt-4o"
    assert config.logging.level == "DEBUG"


def test_app_config_given_missing_logging_defaults_to_info():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.logging.level == "INFO"
