"""Hermetic wiring tests for create_app(): build_registries -> CompletionService -> routes."""

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
        "default_llm": "openai/gpt-4o",
        **agent_overrides,
    }
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}], "agents": [agent]}))
    return config_path


def test_create_app_given_agent_serves_run_with_prepended_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

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

    response = client.post(
        "/v1/agents/missing", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "agent_not_found"


def test_create_app_given_request_tools_reaches_litellm_as_function_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)
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
                    }
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher",
        json={"messages": [{"role": "user", "content": "hi"}], "tools": ["get_current_time"]},
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

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher",
        json={"messages": [{"role": "user", "content": "hi"}], "tools": []},
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert "tools" not in kwargs


def test_create_app_given_inbound_tool_calls_returns_400_and_never_calls_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher",
        json={
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "what did you find?"},
            ]
        },
    )

    assert response.status_code == 400
    mock_acompletion.assert_not_called()


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
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 512,
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 512


def test_create_app_given_valid_config_lists_agents_tools_and_llms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
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

    app = create_app(config_path)
    client = TestClient(app)

    assert client.get("/v1/agents").json() == [
        {"name": "researcher", "default_llm": "openai/gpt-4o", "tools": ["get_current_time"]}
    ]
    assert client.get("/v1/tools").json()[0]["name"] == "get_current_time"
    assert client.get("/v1/llms").json() == ["openai/gpt-4o"]
