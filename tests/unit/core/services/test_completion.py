import pytest

from agent.core.exceptions import AgentNotFoundError, LLMNotFoundError, ToolNotFoundError
from agent.core.models.completion import Completion
from agent.core.models.config import AgentConfig
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.services.completion import CompletionService
from agent.core.tools.get_current_time import GetCurrentTimeTool


class _FakeLLM:
    def __init__(self, completion: Completion) -> None:
        self._completion = completion
        self.last_messages: list[Message] | None = None
        self.last_temperature: float | None = None
        self.last_top_p: float | None = None
        self.last_max_tokens: int | None = None
        self.last_tools: list[dict[str, object]] | None = None

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> Completion:
        self.last_messages = messages
        self.last_temperature = temperature
        self.last_top_p = top_p
        self.last_max_tokens = max_tokens
        self.last_tools = tools
        return self._completion


def _completion() -> Completion:
    return Completion(
        message=Message(role="assistant", content="hi there"),
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        finish_reason="stop",
    )


def _researcher_agent(**overrides: object) -> AgentConfig:
    return AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        default_llm="openai/gpt-4o",
        **overrides,
    )


def _tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("get_current_time", GetCurrentTimeTool())
    return registry


def _service(
    llm: _FakeLLM,
    *,
    agent: AgentConfig | None = None,
    tools: ToolRegistry | None = None,
) -> CompletionService:
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", agent if agent is not None else _researcher_agent())
    return CompletionService(
        llm_registry, agent_registry, tools if tools is not None else ToolRegistry()
    )


async def test_run_given_agent_uses_its_default_llm_and_prepends_system_prompt():
    llm = _FakeLLM(_completion())
    service = _service(llm)
    messages = [Message(role="user", content="hello")]

    run = await service.run(messages, "researcher")

    assert run.model == "openai/gpt-4o"
    assert llm.last_messages is not None
    assert llm.last_messages[0] == Message(role="system", content="You are a research assistant.")
    assert llm.last_messages[1:] == messages


async def test_run_given_model_override_uses_model_with_agent_system_prompt():
    llm = _FakeLLM(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("anthropic/claude-sonnet-5", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = CompletionService(llm_registry, agent_registry, ToolRegistry())
    messages = [Message(role="user", content="hello")]

    run = await service.run(messages, "researcher", model="anthropic/claude-sonnet-5")

    assert run.model == "anthropic/claude-sonnet-5"
    assert llm.last_messages is not None
    assert llm.last_messages[0].content == "You are a research assistant."


async def test_run_given_client_system_message_is_not_merged_or_overridden():
    llm = _FakeLLM(_completion())
    service = _service(llm)
    messages = [
        Message(role="system", content="client's own prompt"),
        Message(role="user", content="hi"),
    ]

    await service.run(messages, "researcher")

    assert llm.last_messages == [
        Message(role="system", content="You are a research assistant."),
        Message(role="system", content="client's own prompt"),
        Message(role="user", content="hi"),
    ]


async def test_run_given_unregistered_agent_raises_agent_not_found_error():
    service = CompletionService(LLMRegistry(), AgentRegistry(), ToolRegistry())

    with pytest.raises(AgentNotFoundError):
        await service.run([Message(role="user", content="hi")], "missing")


async def test_run_given_model_override_to_unregistered_model_raises_llm_not_found_error():
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = CompletionService(LLMRegistry(), agent_registry, ToolRegistry())

    with pytest.raises(LLMNotFoundError):
        await service.run([Message(role="user", content="hi")], "researcher", model="missing/model")


async def test_run_given_request_temperature_overrides_agent_temperature():
    llm = _FakeLLM(_completion())
    service = _service(llm, agent=_researcher_agent(temperature=0.1))

    await service.run([Message(role="user", content="hi")], "researcher", temperature=0.9)

    assert llm.last_temperature == 0.9


async def test_run_given_no_request_temperature_uses_agent_temperature():
    llm = _FakeLLM(_completion())
    service = _service(llm, agent=_researcher_agent(temperature=0.1))

    await service.run([Message(role="user", content="hi")], "researcher")

    assert llm.last_temperature == 0.1


async def test_run_given_no_request_tools_uses_agent_tools():
    llm = _FakeLLM(_completion())
    service = _service(
        llm, agent=_researcher_agent(tools=["get_current_time"]), tools=_tool_registry()
    )

    await service.run([Message(role="user", content="hi")], "researcher")

    assert llm.last_tools is not None
    assert llm.last_tools[0]["function"]["name"] == "get_current_time"


async def test_run_given_empty_request_tools_suppresses_agent_tools():
    llm = _FakeLLM(_completion())
    service = _service(
        llm, agent=_researcher_agent(tools=["get_current_time"]), tools=_tool_registry()
    )

    await service.run([Message(role="user", content="hi")], "researcher", tools=[])

    assert llm.last_tools is None


async def test_run_given_request_tools_overrides_agent_tools():
    llm = _FakeLLM(_completion())
    service = _service(llm, agent=_researcher_agent(tools=[]), tools=_tool_registry())

    await service.run(
        [Message(role="user", content="hi")], "researcher", tools=["get_current_time"]
    )

    assert llm.last_tools is not None
    assert llm.last_tools[0]["function"]["name"] == "get_current_time"


async def test_run_given_unresolvable_tool_name_raises_tool_not_found_error():
    llm = _FakeLLM(_completion())
    service = _service(llm)

    with pytest.raises(ToolNotFoundError):
        await service.run([Message(role="user", content="hi")], "researcher", tools=["missing"])


async def test_run_given_duplicate_request_tools_deduplicates_schemas():
    llm = _FakeLLM(_completion())
    service = _service(llm, tools=_tool_registry())

    await service.run(
        [Message(role="user", content="hi")],
        "researcher",
        tools=["get_current_time", "get_current_time"],
    )

    assert llm.last_tools is not None
    assert len(llm.last_tools) == 1
