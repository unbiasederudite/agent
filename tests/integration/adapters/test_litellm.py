from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import LLMError, LLMRateLimitedError, LLMTimeoutError
from agent.core.models.message import Message, ToolCall, ToolCallFunction


class _FakeProviderError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _fake_litellm_response(finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="hello!"), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def test_complete_given_successful_response_returns_completion(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=_fake_litellm_response()))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.role == "assistant"
    assert completion.message.content == "hello!"
    assert completion.usage.total_tokens == 15
    assert completion.finish_reason == "stop"


async def test_complete_given_length_finish_reason_is_not_hardcoded_to_stop(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(return_value=_fake_litellm_response(finish_reason="length")),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.finish_reason == "length"


async def test_complete_given_litellm_raises_wraps_as_llm_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.acompletion", AsyncMock(side_effect=RuntimeError("provider down")))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_empty_choices_raises_llm_error(monkeypatch: pytest.MonkeyPatch):
    response = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_none_content_raises_llm_error(monkeypatch: pytest.MonkeyPatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_status_code_429_raises_llm_rate_limited_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(side_effect=_FakeProviderError("rate limited", status_code=429)),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMRateLimitedError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_status_code_408_raises_llm_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(side_effect=_FakeProviderError("timed out", status_code=408)),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMTimeoutError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_no_params_and_no_defaults_omits_sampling_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "max_completion_tokens" not in kwargs


async def test_complete_given_constructed_defaults_forwards_them(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", temperature=0.2, top_p=0.9, max_tokens=512)

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_completion_tokens"] == 512


async def test_complete_given_call_value_overrides_constructed_default(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", temperature=0.2)

    await adapter.complete([Message(role="user", content="hi")], temperature=0.9)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.9


async def test_complete_given_call_value_zero_overrides_constructed_default(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o", temperature=0.5)

    await adapter.complete([Message(role="user", content="hi")], temperature=0.0)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["temperature"] == 0.0


async def test_complete_given_tool_calls_in_response_maps_them_and_allows_none_content(
    monkeypatch: pytest.MonkeyPatch,
):
    response = SimpleNamespace(
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
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    completion = await adapter.complete([Message(role="user", content="hi")])

    assert completion.message.content is None
    assert completion.message.tool_calls == [
        ToolCall(id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}"))
    ]
    assert completion.finish_reason == "tool_calls"


async def test_complete_given_tools_param_forwards_to_litellm(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    tools = [
        {
            "type": "function",
            "function": {"name": "get_current_time", "description": "...", "parameters": {}},
        }
    ]

    await adapter.complete([Message(role="user", content="hi")], tools=tools)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["tools"] == tools


async def test_complete_given_no_tools_omits_tools_kwarg(monkeypatch: pytest.MonkeyPatch):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert "tools" not in kwargs


async def test_complete_excludes_none_fields_from_outbound_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete([Message(role="system", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "hi"}


async def test_complete_given_outbound_tool_calls_serializes_nested_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
            )
        ],
    )

    await adapter.complete([message, Message(role="user", content="hi")])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"][0] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_current_time", "arguments": "{}"},
            }
        ],
    }
