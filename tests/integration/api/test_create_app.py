"""Hermetic wiring tests for create_app(): build_registries -> AgentRunService -> routes."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent.api.app import create_app


def _fake_litellm_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello!"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _agent_config_path(tmp_path: Path, **agent_overrides: object) -> Path:
    config_path = tmp_path / "app_config.json"
    agent = {
        "name": "researcher",
        "system_prompt": "You are a research assistant.",
        "model": "openai/gpt-4o",
        "strategy": "react",
        **agent_overrides,
    }
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [agent],
            }
        )
    )
    return config_path


def test_create_app_given_agent_serves_run_with_prepended_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["message"]["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 15

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "You are a research assistant."}


def test_create_app_given_unknown_agent_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/missing", json={"message": "hi"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "agent_not_found"


def test_create_app_given_request_tools_reaches_litellm_as_function_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher", json={"message": "hi", "tools": ["get_current_time"]}
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["tools"][0]["function"]["name"] == "get_current_time"


def test_create_app_given_empty_tools_list_suppresses_agent_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi", "tools": []})

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert "tools" not in kwargs


def test_create_app_given_sampling_params_forwards_them_to_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher",
        json={"message": "hi", "temperature": 0.2, "top_p": 0.9, "max_tokens": 512},
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 512


def test_create_app_given_unknown_strategy_override_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi", "strategy": "rewoo"})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "strategy_not_found"


def test_create_app_given_llm_requests_a_tool_call_executes_it_and_returns_final_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tool_call_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="get_current_time", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_acompletion = AsyncMock(side_effect=[tool_call_response, _fake_litellm_response()])
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "clock-bot",
                        "system_prompt": "You have a clock tool.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/clock-bot", json={"message": "what time is it?"})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["content"] == "hello!"
    assert body["message"]["tool_calls"] is None
    assert body["finish_reason"] == "stop"
    assert mock_acompletion.call_count == 2
    second_call_messages = mock_acompletion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "call_1"
    assert second_call_messages[-1]["name"] == "get_current_time"


def test_create_app_given_valid_config_lists_agents_tools_llms_and_strategies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "tools": [{"name": "get_current_time"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                        "tools": ["get_current_time"],
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    assert client.get("/v1/agents").json() == [
        {
            "name": "researcher",
            "model": "openai/gpt-4o",
            "strategy": "react",
            "tools": ["get_current_time"],
        }
    ]
    assert client.get("/v1/tools").json()[0]["name"] == "get_current_time"
    assert client.get("/v1/models").json() == ["openai/gpt-4o"]
    assert client.get("/v1/strategies").json() == ["react"]
