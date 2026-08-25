import pytest

from agent.core.exceptions import (
    AgentNotFoundError,
    LLMNotFoundError,
    StrategyNotFoundError,
    ToolNotFoundError,
)
from agent.core.models.completion import Completion
from agent.core.models.config import AgentConfig
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.protocols.itool import ITool
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.services.agent_run import AgentRunService
from agent.core.tools.get_current_time import GetCurrentTimeTool


class _FakeStrategy:
    def __init__(self, completion: Completion) -> None:
        self._completion = completion
        self.last_messages: list[Message] | None = None
        self.last_llm: object | None = None
        self.last_tools: dict[str, ITool] | None = None
        self.last_max_iterations: int | None = None
        self.last_temperature: float | None = None
        self.last_top_p: float | None = None
        self.last_max_tokens: int | None = None

    async def run(
        self,
        messages: list[Message],
        llm: object,
        tools: dict[str, ITool],
        max_iterations: int,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self.last_messages = messages
        self.last_llm = llm
        self.last_tools = tools
        self.last_max_iterations = max_iterations
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


def _researcher_agent(**overrides: object) -> AgentConfig:
    return AgentConfig(
        name="researcher",
        system_prompt="You are a research assistant.",
        model="openai/gpt-4o",
        strategy="react",
        **overrides,
    )


def _tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("get_current_time", GetCurrentTimeTool())
    return registry


def _service(
    strategy: object,
    *,
    agent: AgentConfig | None = None,
    tools: ToolRegistry | None = None,
    llm: object = object(),
    base_prompt: str | None = None,
) -> AgentRunService:
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", agent if agent is not None else _researcher_agent())
    strategy_registry = StrategyRegistry()
    strategy_registry.register("react", strategy)
    return AgentRunService(
        llm_registry,
        agent_registry,
        tools if tools is not None else ToolRegistry(),
        strategy_registry,
        base_prompt,
    )


async def test_run_given_agent_uses_its_model_and_prepends_system_prompt():
    strategy = _FakeStrategy(_completion())
    llm = object()
    tools = ToolRegistry()
    service = _service(strategy, llm=llm, tools=tools)

    run = await service.run("hello", "researcher")

    assert run.model == "openai/gpt-4o"
    assert strategy.last_llm is llm
    assert strategy.last_messages == [
        Message(role="system", content="You are a research assistant."),
        Message(role="user", content="hello"),
    ]
    assert strategy.last_tools == {}


async def test_run_given_base_prompt_prepends_it_to_agent_system_prompt():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, base_prompt="House style: be concise.")

    await service.run("hello", "researcher")

    assert strategy.last_messages == [
        Message(
            role="system",
            content="House style: be concise.\n\nYou are a research assistant.",
        ),
        Message(role="user", content="hello"),
    ]


async def test_run_given_model_override_uses_model():
    strategy = _FakeStrategy(_completion())
    llm = object()
    llm_registry = LLMRegistry()
    llm_registry.register("anthropic/claude-sonnet-5", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    strategy_registry = StrategyRegistry()
    strategy_registry.register("react", strategy)
    service = AgentRunService(llm_registry, agent_registry, ToolRegistry(), strategy_registry)

    run = await service.run("hello", "researcher", model="anthropic/claude-sonnet-5")

    assert run.model == "anthropic/claude-sonnet-5"
    assert strategy.last_llm is llm


async def test_run_given_unregistered_agent_raises_agent_not_found_error():
    service = AgentRunService(LLMRegistry(), AgentRegistry(), ToolRegistry(), StrategyRegistry())

    with pytest.raises(AgentNotFoundError):
        await service.run("hi", "missing")


async def test_run_given_model_override_to_unregistered_model_raises_llm_not_found_error():
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = AgentRunService(LLMRegistry(), agent_registry, ToolRegistry(), StrategyRegistry())

    with pytest.raises(LLMNotFoundError):
        await service.run("hi", "researcher", model="missing/model")


async def test_run_given_strategy_override_uses_it_instead_of_agent_strategy():
    default_strategy = _FakeStrategy(_completion())
    override_strategy = _FakeStrategy(_completion())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", object())
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    strategy_registry = StrategyRegistry()
    strategy_registry.register("react", default_strategy)
    strategy_registry.register("rewoo", override_strategy)
    service = AgentRunService(llm_registry, agent_registry, ToolRegistry(), strategy_registry)

    await service.run("hi", "researcher", strategy="rewoo")

    assert override_strategy.last_messages is not None
    assert default_strategy.last_messages is None


async def test_run_given_no_strategy_override_uses_agent_strategy():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent())

    await service.run("hi", "researcher")

    assert strategy.last_messages is not None


async def test_run_given_strategy_override_to_unregistered_strategy_raises_strategy_not_found_error():  # noqa: E501
    service = _service(_FakeStrategy(_completion()))

    with pytest.raises(StrategyNotFoundError):
        await service.run("hi", "researcher", strategy="missing")


async def test_run_given_request_temperature_overrides_agent_temperature():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent(temperature=0.1))

    await service.run("hi", "researcher", temperature=0.9)

    assert strategy.last_temperature == 0.9


async def test_run_given_no_request_temperature_uses_agent_temperature():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent(temperature=0.1))

    await service.run("hi", "researcher")

    assert strategy.last_temperature == 0.1


async def test_run_given_no_request_tools_uses_agent_tools():
    strategy = _FakeStrategy(_completion())
    service = _service(
        strategy, agent=_researcher_agent(tools=["get_current_time"]), tools=_tool_registry()
    )

    await service.run("hi", "researcher")

    assert strategy.last_tools is not None
    assert list(strategy.last_tools) == ["get_current_time"]


async def test_run_given_empty_request_tools_suppresses_agent_tools():
    strategy = _FakeStrategy(_completion())
    service = _service(
        strategy, agent=_researcher_agent(tools=["get_current_time"]), tools=_tool_registry()
    )

    await service.run("hi", "researcher", tools=[])

    assert strategy.last_tools == {}


async def test_run_given_request_tools_overrides_agent_tools():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent(tools=[]), tools=_tool_registry())

    await service.run("hi", "researcher", tools=["get_current_time"])

    assert strategy.last_tools is not None
    assert list(strategy.last_tools) == ["get_current_time"]


async def test_run_given_duplicate_request_tools_resolves_to_one_tool():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent(tools=[]), tools=_tool_registry())

    await service.run("hi", "researcher", tools=["get_current_time", "get_current_time"])

    assert strategy.last_tools is not None
    assert list(strategy.last_tools) == ["get_current_time"]


async def test_run_given_unregistered_tool_raises_tool_not_found_error():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent(tools=["missing"]))

    with pytest.raises(ToolNotFoundError):
        await service.run("hi", "researcher")


async def test_run_passes_agent_max_tool_iterations_to_strategy():
    strategy = _FakeStrategy(_completion())
    service = _service(strategy, agent=_researcher_agent(max_tool_iterations=3))

    await service.run("hi", "researcher")

    assert strategy.last_max_iterations == 3


async def test_run_builds_run_from_strategys_completion():
    completion = Completion(
        message=Message(role="assistant", content="the answer"),
        usage=Usage(prompt_tokens=7, completion_tokens=4, total_tokens=11),
        finish_reason="stop",
    )
    service = _service(_FakeStrategy(completion))

    run = await service.run("hi", "researcher")

    assert run.response == completion.message
    assert run.usage == completion.usage
    assert run.finish_reason == "stop"
