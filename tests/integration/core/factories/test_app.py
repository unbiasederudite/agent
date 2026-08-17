import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.core.exceptions import ConfigError
from agent.core.factories.app import build_registries
from agent.core.models.message import Message


def test_build_given_valid_config_registers_llm(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    llm_registry, agent_registry, tool_registry = build_registries(config_path)

    assert llm_registry.get("openai/gpt-4o") is not None
    assert agent_registry is not None
    assert tool_registry is not None


def test_build_given_duplicate_model_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps({"llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o"}]})
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_missing_file_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "does_not_exist.json"

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_invalid_json_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text("not json")

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_valid_agent_registers_agent(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "default_llm": "openai/gpt-4o",
                    }
                ],
            }
        )
    )

    _, agent_registry, _ = build_registries(config_path)

    assert agent_registry.get("researcher").default_llm == "openai/gpt-4o"


def test_build_given_duplicate_agent_name_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "agents": [
                    {"name": "researcher", "system_prompt": "a", "default_llm": "openai/gpt-4o"},
                    {"name": "researcher", "system_prompt": "b", "default_llm": "openai/gpt-4o"},
                ],
            }
        )
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_agent_unknown_default_llm_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "default_llm": "openai/does-not-exist",
                    }
                ],
            }
        )
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


async def test_build_given_llm_sampling_defaults_wires_adapter_with_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o", "temperature": 0.2}]}))

    llm_registry, _, _ = build_registries(config_path)
    await llm_registry.get("openai/gpt-4o").complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2


def test_build_given_valid_tool_registers_tool(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps({"llms": [{"model": "openai/gpt-4o"}], "tools": [{"name": "get_current_time"}]})
    )

    _, _, tool_registry = build_registries(config_path)

    assert tool_registry.get("get_current_time") is not None


def test_build_given_duplicate_tool_name_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}, {"name": "get_current_time"}],
            }
        )
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_unknown_tool_implementation_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps({"llms": [{"model": "openai/gpt-4o"}], "tools": [{"name": "does-not-exist"}]})
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_agent_unknown_tool_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "default_llm": "openai/gpt-4o",
                        "tools": ["does-not-exist"],
                    }
                ],
            }
        )
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_agent_declared_tool_registers_agent(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "default_llm": "openai/gpt-4o",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    _, agent_registry, _ = build_registries(config_path)

    assert agent_registry.get("researcher").tools == ["get_current_time"]


def test_build_given_agent_duplicate_tool_raises_config_error(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "default_llm": "openai/gpt-4o",
                        "tools": ["get_current_time", "get_current_time"],
                    }
                ],
            }
        )
    )

    with pytest.raises(ConfigError):
        build_registries(config_path)


def test_build_given_two_agents_share_a_tool_registers_both(tmp_path: Path):
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "default_llm": "openai/gpt-4o",
                        "tools": ["get_current_time"],
                    },
                    {
                        "name": "scheduler",
                        "system_prompt": "You are a scheduling assistant.",
                        "default_llm": "openai/gpt-4o",
                        "tools": ["get_current_time"],
                    },
                ],
            }
        )
    )

    _, agent_registry, _ = build_registries(config_path)

    assert agent_registry.get("researcher").tools == ["get_current_time"]
    assert agent_registry.get("scheduler").tools == ["get_current_time"]
