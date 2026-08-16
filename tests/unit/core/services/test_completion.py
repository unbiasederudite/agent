import pytest

from agent.core.exceptions import LLMNotFoundError
from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.registries.llm import LLMRegistry
from agent.core.services.completion import CompletionService


class _FakeLLM:
    def __init__(self, completion: Completion) -> None:
        self._completion = completion

    async def complete(self, messages: list[Message]) -> Completion:
        return self._completion


async def test_run_given_registered_model_returns_run():
    completion = Completion(
        message=Message(role="assistant", content="hi there"),
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        finish_reason="stop",
    )
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", _FakeLLM(completion))
    service = CompletionService(registry)
    messages = [Message(role="user", content="hello")]

    run = await service.run("openai/gpt-4o", messages)

    assert run.model == "openai/gpt-4o"
    assert run.request == messages
    assert run.response == completion.message
    assert run.usage == completion.usage
    assert run.finish_reason == completion.finish_reason


async def test_run_given_unregistered_model_raises_llm_not_found_error():
    registry = LLMRegistry()
    service = CompletionService(registry)

    with pytest.raises(LLMNotFoundError):
        await service.run("missing/model", [Message(role="user", content="hi")])
