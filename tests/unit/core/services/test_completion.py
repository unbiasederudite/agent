import pytest

from agent.core.exceptions import AgentError, AgentNotFoundError, LLMNotFoundError
from agent.core.models.completion import Completion
from agent.core.models.config import AgentConfig
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.services.completion import CompletionService


class _FakeLLM:
    def __init__(self, completion: Completion) -> None:
        self._completion = completion
        self.last_messages: list[Message] | None = None
        self.last_temperature: float | None = None
        self.last_top_p: float | None = None
        self.last_max_tokens: int | None = None

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self.last_messages = messages
        self.last_temperature = temperature
        self.last_top_p = top_p
        self.last_max_tokens = max_tokens
        return self._completion


def _completion() -> Completion:
    return Completion(
        message=Message(role="assistant", content="hi there"),
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        finish_reason="stop",
    )


def _researcher_agent(**overrides: float | int | None) -> AgentConfig:
    return AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        default_llm="openai/gpt-4o",
        **overrides,
    )


async def test_run_given_model_only_uses_model_and_no_system_prompt():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    service = CompletionService(llm_registry, AgentRegistry())
    messages = [Message(role="user", content="hello")]

    run = await service.run(messages, model="openai/gpt-4o")

    assert run.model == "openai/gpt-4o"
    assert run.request == messages
    assert llm.last_messages == messages


async def test_run_given_agent_only_uses_agent_default_llm_and_prepends_system_prompt():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = CompletionService(llm_registry, agent_registry)
    messages = [Message(role="user", content="hello")]

    run = await service.run(messages, agent="researcher")

    assert run.model == "openai/gpt-4o"
    assert llm.last_messages is not None
    assert llm.last_messages[0] == Message(role="system", content="You are a research assistant.")
    assert llm.last_messages[1:] == messages


async def test_run_given_agent_and_model_uses_model_with_agent_system_prompt():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("anthropic/claude-sonnet-5", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = CompletionService(llm_registry, agent_registry)
    messages = [Message(role="user", content="hello")]

    run = await service.run(messages, agent="researcher", model="anthropic/claude-sonnet-5")

    assert run.model == "anthropic/claude-sonnet-5"
    assert llm.last_messages is not None
    assert llm.last_messages[0].content == "You are a research assistant."


async def test_run_given_client_system_message_is_not_merged_or_overridden():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = CompletionService(llm_registry, agent_registry)
    messages = [
        Message(role="system", content="client's own prompt"),
        Message(role="user", content="hi"),
    ]

    await service.run(messages, agent="researcher")

    assert llm.last_messages == [
        Message(role="system", content="You are a research assistant."),
        Message(role="system", content="client's own prompt"),
        Message(role="user", content="hi"),
    ]


async def test_run_given_unregistered_agent_raises_agent_not_found_error():
    service = CompletionService(LLMRegistry(), AgentRegistry())

    with pytest.raises(AgentNotFoundError):
        await service.run([Message(role="user", content="hi")], agent="missing")


async def test_run_given_unregistered_model_raises_llm_not_found_error():
    service = CompletionService(LLMRegistry(), AgentRegistry())

    with pytest.raises(LLMNotFoundError):
        await service.run([Message(role="user", content="hi")], model="missing/model")


async def test_run_given_request_temperature_overrides_agent_temperature():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent(temperature=0.1))
    service = CompletionService(llm_registry, agent_registry)

    await service.run([Message(role="user", content="hi")], agent="researcher", temperature=0.9)

    assert llm.last_temperature == 0.9


async def test_run_given_no_request_temperature_uses_agent_temperature():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent(temperature=0.1))
    service = CompletionService(llm_registry, agent_registry)

    await service.run([Message(role="user", content="hi")], agent="researcher")

    assert llm.last_temperature == 0.1


async def test_run_given_no_agent_and_no_request_sampling_forwards_none():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    service = CompletionService(llm_registry, AgentRegistry())

    await service.run([Message(role="user", content="hi")], model="openai/gpt-4o")

    assert llm.last_temperature is None
    assert llm.last_top_p is None
    assert llm.last_max_tokens is None


async def test_run_given_neither_agent_nor_model_raises_agent_error():
    service = CompletionService(LLMRegistry(), AgentRegistry())

    with pytest.raises(AgentError):
        await service.run([Message(role="user", content="hi")])
