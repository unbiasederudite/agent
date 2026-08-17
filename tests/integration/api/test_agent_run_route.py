from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.app import add_agent_run_route, add_exception_handlers
from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitedError,
    LLMTimeoutError,
    ToolNotFoundError,
)
from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.models.run import Run
from agent.core.models.usage import Usage
from agent.core.services.completion import CompletionService


class _StubCompletionService(CompletionService):
    def __init__(self, result: Run | Exception) -> None:
        self._result = result
        self.last_agent: str | None = None

    async def run(
        self,
        messages: list[Message],
        agent: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
    ) -> Run:
        self.last_agent = agent
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _run(finish_reason: str = "stop") -> Run:
    return Run(
        model="openai/gpt-4o",
        request=[Message(role="user", content="hi")],
        response=Message(role="assistant", content="hello!"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason=finish_reason,
    )


def _client_for(result: Run | Exception) -> TestClient:
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, _StubCompletionService(result))
    return TestClient(app, raise_server_exceptions=False)


def test_run_agent_given_success_returns_200_with_message():
    client = _client_for(_run())

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o"
    assert body["message"]["content"] == "hello!"
    assert body["usage"]["total_tokens"] == 2
    assert body["finish_reason"] == "stop"
    assert body["message"]["tool_calls"] is None


def test_run_agent_uses_the_path_segment_as_the_agent_name():
    service = _StubCompletionService(_run())
    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, service)
    client = TestClient(app)

    client.post("/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]})

    assert service.last_agent == "researcher"


def test_run_agent_given_non_stop_finish_reason_is_not_hardcoded():
    client = _client_for(_run(finish_reason="length"))

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.json()["finish_reason"] == "length"


def test_run_agent_given_unknown_agent_returns_404():
    client = _client_for(AgentNotFoundError("nope"))

    response = client.post(
        "/v1/agents/nope", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["message"] == "nope"
    assert detail["code"] == "agent_not_found"


def test_run_agent_given_unknown_model_override_returns_404():
    client = _client_for(LLMNotFoundError("nope/nope"))

    response = client.post(
        "/v1/agents/researcher",
        json={"messages": [{"role": "user", "content": "hi"}], "model": "nope/nope"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "model_not_found"


def test_run_agent_given_unknown_tool_returns_404():
    client = _client_for(ToolNotFoundError("nope"))

    response = client.post(
        "/v1/agents/researcher",
        json={"messages": [{"role": "user", "content": "hi"}], "tools": ["nope"]},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "tool_not_found"


def test_run_agent_given_llm_failure_returns_502():
    client = _client_for(LLMError("provider down"))

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 502
    assert response.json()["detail"]["message"] == "provider down"


def test_run_agent_given_rate_limited_error_returns_429():
    client = _client_for(LLMRateLimitedError("rate limited"))

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 429


def test_run_agent_given_timeout_error_returns_504():
    client = _client_for(LLMTimeoutError("timed out"))

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 504


def test_run_agent_given_generic_agent_error_returns_500():
    client = _client_for(AgentError("something unexpected"))

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "something unexpected"


def test_run_agent_given_missing_messages_field_returns_400():
    client = _client_for(_run())

    response = client.post("/v1/agents/researcher", json={})

    assert response.status_code == 400
    assert "param" in response.json()["detail"]


def test_run_agent_given_message_with_neither_content_nor_tool_calls_returns_400():
    client = _client_for(_run())

    response = client.post("/v1/agents/researcher", json={"messages": [{"role": "user"}]})

    assert response.status_code == 400
    message = response.json()["detail"]["message"]
    assert message == "either `content` or `tool_calls` must be given"
    assert not message.startswith("Value error,")


def test_run_agent_given_inbound_tool_calls_returns_400():
    client = _client_for(_run())

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
                }
            ]
        },
    )

    assert response.status_code == 400
    message = response.json()["detail"]["message"]
    assert message == "`tool_calls` on a request message is not supported"
    assert not message.startswith("Value error,")


def test_run_agent_given_unexpected_exception_returns_500():
    client = _client_for(RuntimeError("boom"))

    response = client.post(
        "/v1/agents/researcher", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "boom"


def test_run_agent_given_tools_field_returns_tool_calls_passthrough():
    run = Run(
        model="openai/gpt-4o",
        request=[Message(role="user", content="what time is it?")],
        response=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
                )
            ],
        ),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="tool_calls",
    )
    client = _client_for(run)

    response = client.post(
        "/v1/agents/researcher",
        json={
            "messages": [{"role": "user", "content": "what time is it?"}],
            "tools": ["get_current_time"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["finish_reason"] == "tool_calls"
    assert body["message"]["content"] is None
    assert body["message"]["tool_calls"][0]["function"]["name"] == "get_current_time"


def test_unmatched_route_returns_404():
    client = _client_for(_run())

    response = client.get("/does-not-exist")

    assert response.status_code == 404


def test_wrong_http_method_returns_405():
    client = _client_for(_run())

    response = client.get("/v1/agents/researcher")

    assert response.status_code == 405
