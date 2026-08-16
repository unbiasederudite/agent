import pytest
from pydantic import ValidationError

from agent.core.models.config import AgentConfig, AppConfig, LLMConfig, LoggingConfig


def test_llm_config_given_model_string_constructs():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.model == "openai/gpt-4o"


def test_llm_config_given_no_sampling_params_defaults_to_none():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None


def test_llm_config_given_sampling_params_constructs():
    config = LLMConfig(model="openai/gpt-4o", temperature=0.2, top_p=0.9, max_tokens=512)

    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert config.max_tokens == 512


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


def test_agent_config_given_required_fields_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        default_llm="openai/gpt-4o",
    )

    assert config.name == "researcher"
    assert config.system_prompt == "You are a research assistant."
    assert config.default_llm == "openai/gpt-4o"
    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None


def test_agent_config_given_sampling_params_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        default_llm="openai/gpt-4o",
        temperature=0.1,
        top_p=0.8,
        max_tokens=256,
    )

    assert config.temperature == 0.1
    assert config.top_p == 0.8
    assert config.max_tokens == 256


def test_app_config_given_no_agents_defaults_to_empty_list():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.agents == []


def test_app_config_given_agents_constructs():
    raw = (
        '{"llms": [{"model": "openai/gpt-4o"}], '
        '"agents": [{"name": "researcher", "system_prompt": "You are helpful.", '
        '"default_llm": "openai/gpt-4o"}]}'
    )

    config = AppConfig.model_validate_json(raw)

    assert config.agents[0].name == "researcher"
    assert config.agents[0].default_llm == "openai/gpt-4o"
