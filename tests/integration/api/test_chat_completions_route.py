from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.app import add_chat_completions_route
from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
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

    async def run(
        self,
        messages: list[Message],
        *,
        agent: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Run:
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


def test_create_chat_completion_given_unknown_model_returns_404_with_openai_error_shape():
    client = _client_for(LLMNotFoundError("nope/nope"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope/nope", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["message"] == "nope/nope"
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "model_not_found"


def test_create_chat_completion_given_llm_failure_returns_502_with_openai_error_shape():
    client = _client_for(LLMError("provider down"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["message"] == "provider down"
    assert error["type"] == "api_error"


def test_create_chat_completion_given_rate_limited_error_returns_429_with_openai_error_shape():
    client = _client_for(LLMRateLimitedError("rate limited"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 429
    error = response.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "rate_limit_exceeded"


def test_create_chat_completion_given_timeout_error_returns_504_with_openai_error_shape():
    client = _client_for(LLMTimeoutError("timed out"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 504
    error = response.json()["error"]
    assert error["type"] == "api_error"
    assert error["code"] == "timeout"


def test_create_chat_completion_given_generic_agent_error_returns_500_with_openai_error_shape():
    client = _client_for(AgentError("something unexpected"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["message"] == "something unexpected"
    assert error["type"] == "api_error"


def test_create_chat_completion_given_missing_messages_field_returns_400_with_openai_error_shape():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_create_chat_completion_given_invalid_role_returns_400_with_openai_error_shape():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "bogus", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_create_chat_completion_given_neither_agent_nor_model_returns_400_with_openai_error_shape():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not response.json()["error"]["message"].startswith("Value error,")


def test_create_chat_completion_given_unexpected_exception_returns_500_with_openai_error_shape():
    app = FastAPI()
    add_chat_completions_route(app, _StubCompletionService(RuntimeError("boom")))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "api_error"


def test_create_chat_completion_given_agent_only_omits_model_and_succeeds():
    run = Run(
        model="openai/gpt-4o",
        request=[
            Message(role="system", content="persona"),
            Message(role="user", content="hi"),
        ],
        response=Message(role="assistant", content="hello!"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )
    client = _client_for(run)

    response = client.post(
        "/v1/chat/completions",
        json={"agent": "researcher", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "openai/gpt-4o"


def test_create_chat_completion_given_unknown_agent_returns_404_with_openai_error_shape():
    client = _client_for(AgentNotFoundError("nope"))

    response = client.post(
        "/v1/chat/completions",
        json={"agent": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["message"] == "nope"
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "agent_not_found"


def test_unmatched_route_returns_openai_error_shape():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"


def test_wrong_http_method_returns_openai_error_shape():
    client = _client_for(LLMNotFoundError("unused"))

    response = client.get("/v1/chat/completions")

    assert response.status_code == 405
    error = response.json()["error"]
    assert error["type"] == "api_error"


def test_create_chat_completion_given_sampling_fields_returns_200():
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
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 512,
        },
    )

    assert response.status_code == 200
