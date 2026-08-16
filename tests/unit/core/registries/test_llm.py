import pytest

from agent.core.exceptions import LLMNotFoundError
from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.registries.llm import LLMRegistry


class _FakeLLM:
    async def complete(self, messages: list[Message]) -> Completion:
        raise NotImplementedError


def test_llm_registry_get_given_registered_name_returns_llm():
    registry = LLMRegistry()
    llm = _FakeLLM()
    registry.register("openai/gpt-4o", llm)

    assert registry.get("openai/gpt-4o") is llm


def test_llm_registry_get_given_unregistered_name_raises_llm_not_found_error():
    registry = LLMRegistry()

    with pytest.raises(LLMNotFoundError):
        registry.get("missing/model")
