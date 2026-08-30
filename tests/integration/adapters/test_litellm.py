from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import litellm
import pytest

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
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


def _tool_exchange_history() -> list[Message]:
    """A turn whose assistant message carries `tool_calls`, answered by a role="tool" result."""
    return [
        Message(role="user", content="what time is it?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
                )
            ],
        ),
        Message(role="tool", tool_call_id="call_1", name="get_current_time", content="12:00"),
        Message(role="assistant", content="it is noon"),
    ]


_FOLDED_TOOL_EXCHANGE = (
    "[called tool 'get_current_time' with {}]\n"
    "[tool 'get_current_time' returned: 12:00]\n"
    "it is noon"
)

_A_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {"name": "get_current_time", "description": "...", "parameters": {}},
    }
]


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


async def test_complete_given_empty_string_content_and_no_tool_calls_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # An empty-string (not None) reply with no tool_calls would otherwise be stored verbatim
    # and later rejected by Anthropic/Bedrock ("text content blocks must be non-empty").
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=response))
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        await adapter.complete([Message(role="user", content="hi")])


async def test_complete_given_empty_string_content_alongside_tool_calls_normalizes_to_none(
    monkeypatch: pytest.MonkeyPatch,
):
    # An empty-string (not None) reply accompanying real tool_calls must not be stored as-is:
    # `exclude_none=True` keeps an empty string on a later replay, unlike `None`, which could
    # trip the same empty-text-block rejection this file's `content=None`-only guard prevents.
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
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

    # Declares tools: `tool_calls` only ever reach the wire on a tool-declaring request, since
    # a no-tools request has them folded into plain text first.
    await adapter.complete([message, Message(role="user", content="hi")], tools=_A_TOOL_SCHEMA)

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


async def test_complete_given_no_tools_flattens_tool_exchanges_out_of_outbound_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    # Bedrock's Converse API rejects toolUse/toolResult blocks in a request with no toolConfig,
    # even when they only replay an earlier exchange -- so a no-tools call must never send them.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(_tool_exchange_history())

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": _FOLDED_TOOL_EXCHANGE},
    ]


async def test_complete_given_empty_tools_list_flattens_tool_exchanges_too(
    monkeypatch: pytest.MonkeyPatch,
):
    # `[]` is as much "no tools declared" as `None` is -- the adapter must not rely on callers
    # having normalized one to the other.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(_tool_exchange_history(), tools=[])

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": _FOLDED_TOOL_EXCHANGE},
    ]


async def test_complete_given_no_tools_history_ending_on_tool_result_flattens_to_trailing_assistant(
    monkeypatch: pytest.MonkeyPatch,
):
    # Unlike `_tool_exchange_history()` (which ends on a real assistant reply), history cut off
    # mid-tool-use has nothing after the last tool result -- `flush()` folds it into a brand-new
    # synthetic assistant message with nothing after it. `ReactStrategy`'s forced-final call is
    # the one caller that can reach this shape (see its own docstring); this proves what actually
    # goes out on the wire in that case, which is why that caller appends a trailing instruction.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")
    history = [
        Message(role="user", content="what time is it?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}")
                )
            ],
        ),
        Message(role="tool", tool_call_id="call_1", name="get_current_time", content="12:00"),
    ]

    await adapter.complete(history)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "content": (
                "[called tool 'get_current_time' with {}]\n"
                "[tool 'get_current_time' returned: 12:00]"
            ),
        },
    ]


async def test_complete_given_tools_declared_sends_the_tool_exchange_unflattened(
    monkeypatch: pytest.MonkeyPatch,
):
    # The regression that matters: flattening is for no-tools requests only. A tool-declaring
    # request must still replay the genuine toolUse/toolResult blocks, or the provider loses
    # the pairing between a tool call and its result.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(_tool_exchange_history(), tools=_A_TOOL_SCHEMA)

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "what time is it?"},
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
        {
            "role": "tool",
            "content": "12:00",
            "tool_call_id": "call_1",
            "name": "get_current_time",
        },
        {"role": "assistant", "content": "it is noon"},
    ]


async def test_complete_given_no_tools_and_no_tool_content_leaves_messages_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    # Making the flattening unconditional for no-tools calls costs nothing in the common case.
    mock_acompletion = AsyncMock(return_value=_fake_litellm_response())
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    await adapter.complete(
        [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    )

    _, kwargs = mock_acompletion.call_args
    assert kwargs["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


async def test_complete_given_context_window_exceeded_raises_llm_context_window_exceeded_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "litellm.acompletion",
        AsyncMock(
            side_effect=litellm.ContextWindowExceededError(
                message="context exceeded", model="openai/gpt-4o", llm_provider="openai"
            )
        ),
    )
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMContextWindowExceededError):
        await adapter.complete([Message(role="user", content="hi")])


def test_max_input_tokens_given_known_model_returns_it(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {"max_input_tokens": 128000})
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    assert adapter.max_input_tokens() == 128000


def test_max_input_tokens_given_unrecognized_model_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def _raise(model: str) -> dict[str, int]:
        raise ValueError("model isn't mapped")

    monkeypatch.setattr("litellm.get_model_info", _raise)
    adapter = LiteLLMAdapter(model="openai/does-not-exist")

    with pytest.raises(LLMError):
        adapter.max_input_tokens()


def test_max_input_tokens_given_missing_field_raises_llm_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {})
    adapter = LiteLLMAdapter(model="openai/gpt-4o")

    with pytest.raises(LLMError):
        adapter.max_input_tokens()


def test_max_input_tokens_given_context_window_override_returns_it_without_litellm_lookup(
    monkeypatch: pytest.MonkeyPatch,
):
    mock_get_model_info = Mock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr("litellm.get_model_info", mock_get_model_info)
    adapter = LiteLLMAdapter(model="self-hosted/my-model", context_window=32000)

    assert adapter.max_input_tokens() == 32000
    mock_get_model_info.assert_not_called()


def test_max_input_tokens_given_context_window_override_takes_precedence_over_real_litellm_data(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {"max_input_tokens": 128000})
    adapter = LiteLLMAdapter(model="openai/gpt-4o", context_window=999)

    assert adapter.max_input_tokens() == 999
