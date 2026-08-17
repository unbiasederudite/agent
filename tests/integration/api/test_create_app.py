"""Hermetic wiring tests for create_app(): build_registries -> CompletionService -> route.

Covers both the model-only request path and the agent-only request path.
"""

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


def test_create_app_given_valid_config_serves_chat_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["choices"][0]["message"]["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 15


def test_create_app_given_agent_only_serves_chat_completion_with_prepended_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
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

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"agent": "researcher", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {
        "role": "system",
        "content": "You are a research assistant.",
    }


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
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": ["get_current_time"],
        },
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
        "/v1/chat/completions",
        json={"agent": "researcher", "messages": [{"role": "user", "content": "hi"}], "tools": []},
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert "tools" not in kwargs


def test_create_app_given_inbound_tool_calls_returns_400_and_never_calls_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
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
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    mock_acompletion.assert_not_called()


def test_create_app_given_max_completion_tokens_forwards_it_to_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 512,
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs


def test_create_app_given_unrecognized_extra_field_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs


def test_create_app_given_both_max_tokens_fields_only_current_one_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(json.dumps({"llms": [{"model": "openai/gpt-4o"}]}))

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 999,
            "max_completion_tokens": 512,
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_acompletion.call_args
    assert kwargs["max_completion_tokens"] == 512
