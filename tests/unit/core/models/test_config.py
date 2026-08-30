import pytest
from pydantic import ValidationError

from agent.core.models.config import (
    AgentConfig,
    AppConfig,
    CompactionConfig,
    LLMConfig,
    LoggingConfig,
    ToolConfig,
)


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


def test_llm_config_given_no_context_window_defaults_to_none():
    config = LLMConfig(model="openai/gpt-4o")

    assert config.context_window is None


def test_llm_config_given_context_window_constructs():
    config = LLMConfig(model="openai/gpt-4o", context_window=128000)

    assert config.context_window == 128000


def test_llm_config_given_context_window_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", context_window=0)


def test_llm_config_given_context_window_negative_raises_validation_error():
    with pytest.raises(ValidationError):
        LLMConfig(model="openai/gpt-4o", context_window=-1)


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
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.name == "researcher"
    assert config.system_prompt == "You are a research assistant."
    assert config.model == "openai/gpt-4o"
    assert config.strategy == "react"
    assert config.max_tool_iterations == 10
    assert config.temperature is None
    assert config.top_p is None
    assert config.max_tokens is None


def test_agent_config_given_sampling_params_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
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
        '"model": "openai/gpt-4o", "strategy": "react"}]}'
    )

    config = AppConfig.model_validate_json(raw)

    assert config.agents[0].name == "researcher"
    assert config.agents[0].model == "openai/gpt-4o"


def test_tool_config_given_name_constructs():
    config = ToolConfig(name="get_current_time")

    assert config.name == "get_current_time"


def test_agent_config_given_no_tools_defaults_to_empty_list():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.tools == []


def test_agent_config_given_tools_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        tools=["get_current_time"],
    )

    assert config.tools == ["get_current_time"]


def test_app_config_given_no_tools_defaults_to_empty_list():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.tools == []


def test_agent_config_given_no_strategy_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
        )


def test_agent_config_given_max_tool_iterations_override_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_tool_iterations=3,
    )

    assert config.max_tool_iterations == 3


def test_app_config_given_tools_constructs():
    raw = '{"llms": [{"model": "openai/gpt-4o"}], "tools": [{"name": "get_current_time"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.tools[0].name == "get_current_time"


def test_agent_config_given_no_max_input_chars_defaults_to_none():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.max_input_chars is None


def test_agent_config_given_max_input_chars_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_input_chars=20000,
    )

    assert config.max_input_chars == 20000


def test_agent_config_given_max_input_chars_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_input_chars=0,
        )


def test_agent_config_given_no_max_tool_result_chars_defaults_to_none():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert config.max_tool_result_chars is None


def test_agent_config_given_max_tool_result_chars_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_tool_result_chars=5000,
    )

    assert config.max_tool_result_chars == 5000


def test_agent_config_given_max_tool_result_chars_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_result_chars=0,
        )


def test_agent_config_given_max_tool_result_chars_negative_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_result_chars=-1,
        )


def test_compaction_config_given_required_field_constructs_with_defaults():
    config = CompactionConfig(model="anthropic/claude-3-5-haiku-20241022")

    assert config.model == "anthropic/claude-3-5-haiku-20241022"
    assert config.token_budget_pct == 0.8
    assert config.keep_recent_turns == 4
    assert "Summarize" in config.prompt


def test_compaction_config_given_overrides_constructs():
    config = CompactionConfig(
        model="anthropic/claude-3-5-haiku-20241022",
        token_budget_pct=0.5,
        keep_recent_turns=2,
        prompt="Custom instructions.",
    )

    assert config.token_budget_pct == 0.5
    assert config.keep_recent_turns == 2
    assert config.prompt == "Custom instructions."


def test_compaction_config_given_token_budget_pct_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", token_budget_pct=0.0)


def test_compaction_config_given_token_budget_pct_above_one_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", token_budget_pct=1.1)


def test_compaction_config_given_negative_keep_recent_turns_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", keep_recent_turns=-1)


def test_app_config_given_no_compaction_defaults_to_none():
    raw = '{"llms": [{"model": "openai/gpt-4o"}]}'

    config = AppConfig.model_validate_json(raw)

    assert config.compaction is None


def test_app_config_given_compaction_constructs():
    raw = (
        '{"llms": [{"model": "openai/gpt-4o"}], '
        '"compaction": {"model": "openai/gpt-4o", "token_budget_pct": 0.7, '
        '"keep_recent_turns": 2}}'
    )

    config = AppConfig.model_validate_json(raw)

    assert config.compaction is not None
    assert config.compaction.model == "openai/gpt-4o"
    assert config.compaction.token_budget_pct == 0.7
    assert config.compaction.keep_recent_turns == 2


def test_agent_config_given_no_tool_call_and_total_char_caps_defaults_to_none():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
    )

    assert (config.max_tool_calls_per_round, config.max_tool_results_total_chars) == (None, None)


def test_agent_config_given_tool_call_and_total_char_caps_constructs():
    config = AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        max_tool_calls_per_round=3,
        max_tool_results_total_chars=200_000,
    )

    assert (config.max_tool_calls_per_round, config.max_tool_results_total_chars) == (3, 200_000)


def test_agent_config_given_max_tool_calls_per_round_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_calls_per_round=0,
        )


def test_agent_config_given_max_tool_results_total_chars_zero_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentConfig(
            name="researcher",
            system_prompt="You are a research assistant.",
            model="openai/gpt-4o",
            strategy="react",
            max_tool_results_total_chars=0,
        )


def test_compaction_config_given_no_chunk_turns_defaults_to_four():
    assert CompactionConfig(model="anthropic/claude-3-5-haiku-20241022").chunk_turns == 4


def test_compaction_config_given_chunk_turns_constructs():
    config = CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", chunk_turns=2)

    assert config.chunk_turns == 2


def test_compaction_config_given_zero_chunk_turns_raises_validation_error():
    with pytest.raises(ValidationError):
        CompactionConfig(model="anthropic/claude-3-5-haiku-20241022", chunk_turns=0)
