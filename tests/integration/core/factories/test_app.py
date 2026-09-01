from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.core.exceptions import ConfigError
from agent.core.factories.app import build_registries
from agent.core.models.config import AppConfig
from agent.core.models.message import Message


def _agent(**overrides: object) -> dict[str, object]:
    return {
        "name": "researcher",
        "system_prompt": "You are a research assistant.",
        "model": "openai/gpt-4o",
        "strategy": "react",
        **overrides,
    }


def test_build_given_valid_config_registers_llm():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    (
        llm_registry,
        agent_registry,
        tool_registry,
        strategy_registry,
        base_prompt,
        compaction_config,
        logging_config,
        max_sessions,
    ) = build_registries(config)

    assert llm_registry.get("openai/gpt-4o") is not None
    assert agent_registry is not None
    assert tool_registry is not None
    assert strategy_registry is not None
    assert base_prompt is None
    assert compaction_config is None
    assert logging_config.level == "INFO"


def test_build_given_base_prompt_returns_it():
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o"}], "base_prompt": "House style."}
    )

    _, _, _, _, base_prompt, _, _, _ = build_registries(config)

    assert base_prompt == "House style."


def test_build_given_duplicate_model_raises_config_error():
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o"}]}
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_valid_agent_registers_agent():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent()],
        }
    )

    _, agent_registry, _, _, _, _, _, _ = build_registries(config)

    assert agent_registry.get("researcher").model == "openai/gpt-4o"


def test_build_given_duplicate_agent_name_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent(system_prompt="a"), _agent(system_prompt="b")],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_agent_unknown_model_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent(model="openai/does-not-exist")],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_agent_unknown_strategy_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent(strategy="rewoo")],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


async def test_build_given_llm_sampling_defaults_wires_adapter_with_them(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o", "temperature": 0.2}]})

    llm_registry, _, _, _, _, _, _, _ = build_registries(config)
    await llm_registry.get("openai/gpt-4o").complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2


def test_build_given_context_window_wires_adapter_with_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {"max_input_tokens": 128000})
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o", "context_window": 32000}]}
    )

    llm_registry, _, _, _, _, _, _, _ = build_registries(config)

    assert llm_registry.get("openai/gpt-4o").max_input_tokens() == 32000


def test_build_given_valid_tool_registers_tool():
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o"}], "tools": [{"name": "get_current_time"}]}
    )

    _, _, tool_registry, _, _, _, _, _ = build_registries(config)

    assert tool_registry.get("get_current_time") is not None


def test_build_given_duplicate_tool_name_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "tools": [{"name": "get_current_time"}, {"name": "get_current_time"}],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_unknown_tool_implementation_raises_config_error():
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o"}], "tools": [{"name": "does-not-exist"}]}
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_valid_strategy_registers_strategy():
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o"}], "strategies": [{"name": "react"}]}
    )

    _, _, _, strategy_registry, _, _, _, _ = build_registries(config)

    assert strategy_registry.get("react") is not None


def test_build_given_no_strategies_declared_registers_none():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    _, _, _, strategy_registry, _, _, _, _ = build_registries(config)

    assert strategy_registry.all() == {}


def test_build_given_duplicate_strategy_name_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}, {"name": "react"}],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_unknown_strategy_implementation_raises_config_error():
    config = AppConfig.model_validate(
        {"llms": [{"model": "openai/gpt-4o"}], "strategies": [{"name": "does-not-exist"}]}
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_agent_unknown_tool_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent(tools=["does-not-exist"])],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_agent_declared_tool_registers_agent():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "tools": [{"name": "get_current_time"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent(tools=["get_current_time"])],
        }
    )

    _, agent_registry, _, _, _, _, _, _ = build_registries(config)

    assert agent_registry.get("researcher").tools == ["get_current_time"]


def test_build_given_agent_duplicate_tool_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "tools": [{"name": "get_current_time"}],
            "strategies": [{"name": "react"}],
            "agents": [_agent(tools=["get_current_time", "get_current_time"])],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_two_agents_share_a_tool_registers_both():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "tools": [{"name": "get_current_time"}],
            "strategies": [{"name": "react"}],
            "agents": [
                _agent(tools=["get_current_time"]),
                _agent(
                    name="scheduler",
                    system_prompt="You are a scheduling assistant.",
                    tools=["get_current_time"],
                ),
            ],
        }
    )

    _, agent_registry, _, _, _, _, _, _ = build_registries(config)

    assert agent_registry.get("researcher").tools == ["get_current_time"]
    assert agent_registry.get("scheduler").tools == ["get_current_time"]


def test_build_given_no_compaction_returns_none():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    _, _, _, _, _, compaction_config, _, _ = build_registries(config)

    assert compaction_config is None


def test_build_given_compaction_with_declared_model_returns_it():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "compaction": {"model": "openai/gpt-4o", "token_budget_pct": 0.7},
        }
    )

    _, _, _, _, _, compaction_config, _, _ = build_registries(config)

    assert compaction_config is not None
    assert compaction_config.model == "openai/gpt-4o"
    assert compaction_config.token_budget_pct == 0.7


def test_build_given_compaction_unknown_model_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "compaction": {"model": "openai/does-not-exist"},
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


async def test_build_given_llm_retry_and_timeout_settings_wires_adapter_with_them(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config = AppConfig.model_validate(
        {
            "llms": [
                {
                    "model": "openai/gpt-4o",
                    "num_retries": 0,
                    "timeout": 12.5,
                    "max_concurrent_requests": 3,
                }
            ]
        }
    )

    llm_registry, _, _, _, _, _, _, _ = build_registries(config)
    adapter = llm_registry.get("openai/gpt-4o")
    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["timeout"] == 12.5
    assert adapter._num_retries == 0
    assert adapter._max_concurrent == 3


def test_build_given_logging_config_returns_it():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "logging": {"level": "DEBUG", "format": "json"},
        }
    )

    *_, logging_config, _ = build_registries(config)

    assert logging_config.level == "DEBUG"
    assert logging_config.format == "json"


def test_build_given_no_logging_block_returns_defaults():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    *_, logging_config, _ = build_registries(config)

    assert logging_config.level == "INFO"
    assert logging_config.format == "text"


def test_build_given_max_sessions_returns_it():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}], "max_sessions": 100})

    *_, max_sessions = build_registries(config)

    assert max_sessions == 100


def test_build_given_no_max_sessions_returns_none():
    config = AppConfig.model_validate({"llms": [{"model": "openai/gpt-4o"}]})

    *_, max_sessions = build_registries(config)

    assert max_sessions is None


def test_build_given_agent_allowed_tools_references_unknown_tool_raises_config_error():
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [
                {
                    "name": "researcher",
                    "system_prompt": "You are a research assistant.",
                    "model": "openai/gpt-4o",
                    "strategy": "react",
                    "allowed_tools": ["nonexistent_tool"],
                }
            ],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_agent_allowed_models_references_unknown_model_raises_config_error():
    # allowed_models includes the agent's own model (openai/gpt-4o) so AgentConfig's own
    # _defaults_within_ceilings validator is satisfied at construction time -- the extra
    # "nonexistent_model" entry is what exercises build_registries()'s own cross-reference
    # check (not the agent's own model/ceiling self-consistency, already covered in
    # tests/unit/core/models/test_config.py).
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [
                {
                    "name": "researcher",
                    "system_prompt": "You are a research assistant.",
                    "model": "openai/gpt-4o",
                    "strategy": "react",
                    "allowed_models": ["openai/gpt-4o", "nonexistent_model"],
                }
            ],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)


def test_build_given_agent_allowed_strategies_references_unknown_strategy_raises_config_error():
    # allowed_strategies includes the agent's own strategy (react) for the same reason as
    # the allowed_models test above.
    config = AppConfig.model_validate(
        {
            "llms": [{"model": "openai/gpt-4o"}],
            "strategies": [{"name": "react"}],
            "agents": [
                {
                    "name": "researcher",
                    "system_prompt": "You are a research assistant.",
                    "model": "openai/gpt-4o",
                    "strategy": "react",
                    "allowed_strategies": ["react", "nonexistent_strategy"],
                }
            ],
        }
    )

    with pytest.raises(ConfigError):
        build_registries(config)
