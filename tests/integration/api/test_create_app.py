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


def test_create_app_given_no_session_id_returns_a_new_one_and_second_call_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "my name is Sam"})
    session_id = first.json()["session_id"]
    assert session_id

    client.post(
        "/v1/agents/researcher",
        json={"message": "what's my name?", "session_id": session_id},
    )

    second_call_messages = mock_acompletion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[1] == {"role": "user", "content": "my name is Sam"}
    assert second_call_messages[2] == {"role": "assistant", "content": "hello!"}
    assert second_call_messages[3] == {"role": "user", "content": "what's my name?"}


def test_create_app_given_unknown_session_id_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/researcher", json={"message": "hi", "session_id": "does-not-exist"}
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_create_app_given_session_id_reused_under_a_different_agent_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    },
                    {
                        "name": "scheduler",
                        "system_prompt": "You are a scheduling assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    },
                ],
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "hi"})
    session_id = first.json()["session_id"]

    response = client.post("/v1/agents/scheduler", json={"message": "hi", "session_id": session_id})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_create_app_given_message_exceeds_max_input_chars_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    config_path = _agent_config_path(tmp_path, max_input_chars=5)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "this is too long"})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "input_too_large"


def test_create_app_given_compaction_configured_wires_and_invokes_compaction_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A `compaction` block in config makes create_app() build a real, working service.

    Once recorded usage crosses the tiny configured budget, the next call to the same
    session compacts first, actually calling the configured summarizer.
    """
    summary_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="summary", tool_calls=None), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )
    mock_acompletion = AsyncMock(
        side_effect=[_fake_litellm_response(), summary_response, _fake_litellm_response()]
    )
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o-mini"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "compaction": {
                    "model": "openai/gpt-4o-mini",
                    "token_budget_pct": 0.0001,
                    "keep_recent_turns": 0,
                    "prompt": "Summarize this.",
                },
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)

    first = client.post("/v1/agents/researcher", json={"message": "hi"})
    session_id = first.json()["session_id"]

    second = client.post(
        "/v1/agents/researcher",
        json={"message": "what's next?", "session_id": session_id},
    )

    assert second.status_code == 200
    assert mock_acompletion.call_count == 3
    summarizer_call = mock_acompletion.call_args_list[1]
    assert summarizer_call.kwargs["model"] == "openai/gpt-4o-mini"
    assert summarizer_call.kwargs["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "Summarize this."},
    ]


def test_create_app_given_context_window_exceeded_with_no_compaction_returns_502(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import litellm

    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(
            side_effect=litellm.ContextWindowExceededError(
                message="too big", model="openai/gpt-4o", llm_provider="openai"
            )
        ),
    )
    config_path = _agent_config_path(tmp_path)

    app = create_app(config_path)
    client = TestClient(app)

    response = client.post("/v1/agents/researcher", json={"message": "hi"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "context_window_exceeded"


def test_create_app_given_compaction_cannot_help_returns_502_compaction_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Second call: the agent call overflows, and every compaction tier fails too (the
    # summarizer overflows as well, and a one-turn history can't be chunked) -- the
    # exhausted case, which must map to its own code, not the generic overflow one.
    import litellm

    overflow = litellm.ContextWindowExceededError(
        message="too big", model="openai/gpt-4o", llm_provider="openai"
    )
    mock_acompletion = AsyncMock(side_effect=[_fake_litellm_response(), *[overflow] * 20])
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    config_path = tmp_path / "app_config.json"
    config_path.write_text(
        json.dumps(
            {
                "llms": [{"model": "openai/gpt-4o"}, {"model": "openai/gpt-4o-mini"}],
                "strategies": [{"name": "react"}],
                "agents": [
                    {
                        "name": "researcher",
                        "system_prompt": "You are a research assistant.",
                        "model": "openai/gpt-4o",
                        "strategy": "react",
                    }
                ],
                "compaction": {
                    "model": "openai/gpt-4o-mini",
                    "keep_recent_turns": 1,
                    "prompt": "Summarize this.",
                },
            }
        )
    )

    app = create_app(config_path)
    client = TestClient(app)
    session_id = client.post("/v1/agents/researcher", json={"message": "hi"}).json()["session_id"]

    response = client.post(
        "/v1/agents/researcher",
        json={"message": "what's next?", "session_id": session_id},
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "compaction_exhausted"
