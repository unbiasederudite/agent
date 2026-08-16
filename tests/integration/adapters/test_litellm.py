from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import LLMError, LLMRateLimitedError, LLMTimeoutError
from agent.core.models.message import Message


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
