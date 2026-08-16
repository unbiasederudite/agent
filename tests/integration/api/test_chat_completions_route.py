from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.app import add_chat_completions_route
from agent.core.exceptions import (
    AgentError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.models.usage import Usage
from agent.core.services.completion import CompletionService


class _StubCompletionService(CompletionService):
    def __init__(self, result: Run | Exception) -> None:
        self._result = result

    async def run(self, model: str, messages: list[Message]) -> Run:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _client_for(result: Run | Exception) -> TestClient:
    app = FastAPI()
    add_chat_completions_route(app, _StubCompletionService(result))
    return TestClient(app)


def test_create_chat_completion_given_success_returns_200_with_choice():
    run = Run(
        model="openai/gpt-4o",
        request=[Message(role="user", content="hi")],
        response=Message(role="assistant", content="hello!"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )
    client = _client_for(run)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["choices"][0]["message"]["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 2


def test_create_chat_completion_given_success_returns_expected_envelope_fields():
    run = Run(
        model="openai/gpt-4o",
        request=[Message(role="user", content="hi")],
        response=Message(role="assistant", content="hello!"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )
    client = _client_for(run)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)


def test_create_chat_completion_given_non_stop_finish_reason_is_not_hardcoded():
    run = Run(
        model="openai/gpt-4o",
        request=[Message(role="user", content="hi")],
        response=Message(role="assistant", content="hello!"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="length",
    )
    client = _client_for(run)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_create_chat_completion_given_unknown_model_returns_404():
    client = _client_for(LLMNotFoundError("nope/nope"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope/nope", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 404


def test_create_chat_completion_given_llm_failure_returns_502():
    client = _client_for(LLMError("provider down"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502


def test_create_chat_completion_given_rate_limited_error_returns_429():
    client = _client_for(LLMRateLimitedError("rate limited"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 429


def test_create_chat_completion_given_timeout_error_returns_504():
    client = _client_for(LLMTimeoutError("timed out"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 504


def test_create_chat_completion_given_generic_agent_error_returns_500():
    client = _client_for(AgentError("something unexpected"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 500


def test_create_chat_completion_given_missing_messages_field_returns_422():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o"},
    )

    assert response.status_code == 422


def test_create_chat_completion_given_invalid_role_returns_422():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "bogus", "content": "hi"}]},
    )

    assert response.status_code == 422
