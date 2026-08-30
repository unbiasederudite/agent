import pytest

from agent.core.exceptions import (
    AgentNotFoundError,
    CompactionExhaustedError,
    InputTooLargeError,
    LLMContextWindowExceededError,
    LLMNotFoundError,
    SessionNotFoundError,
    StrategyNotFoundError,
    ToolNotFoundError,
)
from agent.core.models.config import AgentConfig
from agent.core.models.message import Message
from agent.core.models.turn import Turn
from agent.core.models.usage import Usage
from agent.core.protocols.itool import ITool
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.services.agent_run import AgentRunService
from agent.core.session_stores.in_memory import InMemorySessionStore
from agent.core.tools.get_current_time import GetCurrentTimeTool


class _FakeStrategy:
    """Returns queued outcomes in order, one per `run()` call; the last one repeats."""

    def __init__(self, outcome: Turn | list[Turn | Exception]) -> None:
        self._outcomes = outcome if isinstance(outcome, list) else [outcome]
        self._call_count = 0
        self.last_messages: list[Message] | None = None
        self.last_llm: object | None = None
        self.last_tools: dict[str, ITool] | None = None
        self.last_max_iterations: int | None = None
        self.last_temperature: float | None = None
        self.last_top_p: float | None = None
        self.last_max_tokens: int | None = None
        self.last_max_tool_result_chars: int | None = None
        self.last_max_tool_calls_per_round: int | None = None
        self.last_max_tool_results_total_chars: int | None = None
        self.call_messages: list[list[Message]] = []

    async def run(
        self,
        messages: list[Message],
        llm: object,
        tools: dict[str, ITool],
        max_iterations: int,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        max_tool_calls_per_round: int | None = None,
        max_tool_results_total_chars: int | None = None,
    ) -> Turn:
        self.last_messages = messages
        self.last_llm = llm
        self.last_tools = tools
        self.last_max_iterations = max_iterations
        self.last_temperature = temperature
        self.last_top_p = top_p
        self.last_max_tokens = max_tokens
        self.last_max_tool_result_chars = max_tool_result_chars
        self.last_max_tool_calls_per_round = max_tool_calls_per_round
        self.last_max_tool_results_total_chars = max_tool_results_total_chars
        self.call_messages.append(messages)
        index = min(self._call_count, len(self._outcomes) - 1)
        self._call_count += 1
        outcome = self._outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeCompactionService:
    def __init__(
        self,
        compact_result: bool = True,
        session_store: object | None = None,
    ) -> None:
        self.maybe_compact_calls: list[tuple[str, str, str]] = []
        self.record_usage_calls: list[tuple[str, str, int]] = []
        self.compact_calls: list[tuple[str, str]] = []
        self._compact_result = compact_result
        self._session_store = session_store

    async def maybe_compact(self, agent: str, session_id: str, model: str) -> None:
        self.maybe_compact_calls.append((agent, session_id, model))

    def record_usage(self, agent: str, session_id: str, total_tokens: int) -> None:
        self.record_usage_calls.append((agent, session_id, total_tokens))

    async def compact(self, agent: str, session_id: str) -> bool:
        self.compact_calls.append((agent, session_id))
        if self._compact_result and self._session_store is not None:
            # Mimic real CompactionService: history actually changes on a successful pass,
            # so callers that re-fetch after compacting see a different message list.
            await self._session_store.replace(
                agent, session_id, [Message(role="user", content="summary")]
            )
        return self._compact_result


def _turn(content: str = "hi there") -> Turn:
    return Turn(
        messages=[Message(role="assistant", content=content)],
        usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        finish_reason="stop",
        final_total_tokens=5,
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
    session_store: object | None = None,
    compaction_service: object | None = None,
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
        session_store if session_store is not None else InMemorySessionStore(),
        compaction_service,
    )


async def test_run_given_agent_uses_its_model_and_prepends_system_prompt():
    strategy = _FakeStrategy(_turn())
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
    strategy = _FakeStrategy(_turn())
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
    strategy = _FakeStrategy(_turn())
    llm = object()
    llm_registry = LLMRegistry()
    llm_registry.register("anthropic/claude-sonnet-5", llm)
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    strategy_registry = StrategyRegistry()
    strategy_registry.register("react", strategy)
    service = AgentRunService(
        llm_registry,
        agent_registry,
        ToolRegistry(),
        strategy_registry,
        None,
        InMemorySessionStore(),
    )

    run = await service.run("hello", "researcher", model="anthropic/claude-sonnet-5")

    assert run.model == "anthropic/claude-sonnet-5"
    assert strategy.last_llm is llm


async def test_run_given_unregistered_agent_raises_agent_not_found_error():
    service = AgentRunService(
        LLMRegistry(),
        AgentRegistry(),
        ToolRegistry(),
        StrategyRegistry(),
        None,
        InMemorySessionStore(),
    )

    with pytest.raises(AgentNotFoundError):
        await service.run("hi", "missing")


async def test_run_given_model_override_to_unregistered_model_raises_llm_not_found_error():
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    service = AgentRunService(
        LLMRegistry(),
        agent_registry,
        ToolRegistry(),
        StrategyRegistry(),
        None,
        InMemorySessionStore(),
    )

    with pytest.raises(LLMNotFoundError):
        await service.run("hi", "researcher", model="missing/model")


async def test_run_given_strategy_override_uses_it_instead_of_agent_strategy():
    default_strategy = _FakeStrategy(_turn())
    override_strategy = _FakeStrategy(_turn())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", object())
    agent_registry = AgentRegistry()
    agent_registry.register("researcher", _researcher_agent())
    strategy_registry = StrategyRegistry()
    strategy_registry.register("react", default_strategy)
    strategy_registry.register("rewoo", override_strategy)
    service = AgentRunService(
        llm_registry,
        agent_registry,
        ToolRegistry(),
        strategy_registry,
        None,
        InMemorySessionStore(),
    )

    await service.run("hi", "researcher", strategy="rewoo")

    assert override_strategy.last_messages is not None
    assert default_strategy.last_messages is None


async def test_run_given_no_strategy_override_uses_agent_strategy():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent())

    await service.run("hi", "researcher")

    assert strategy.last_messages is not None


async def test_run_given_strategy_override_to_unregistered_strategy_raises_strategy_not_found_error():  # noqa: E501
    service = _service(_FakeStrategy(_turn()))

    with pytest.raises(StrategyNotFoundError):
        await service.run("hi", "researcher", strategy="missing")


async def test_run_given_request_temperature_overrides_agent_temperature():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(temperature=0.1))

    await service.run("hi", "researcher", temperature=0.9)

    assert strategy.last_temperature == 0.9


async def test_run_given_no_request_temperature_uses_agent_temperature():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(temperature=0.1))

    await service.run("hi", "researcher")

    assert strategy.last_temperature == 0.1


async def test_run_given_no_request_tools_uses_agent_tools():
    strategy = _FakeStrategy(_turn())
    service = _service(
        strategy, agent=_researcher_agent(tools=["get_current_time"]), tools=_tool_registry()
    )

    await service.run("hi", "researcher")

    assert strategy.last_tools is not None
    assert list(strategy.last_tools) == ["get_current_time"]


async def test_run_given_empty_request_tools_suppresses_agent_tools():
    strategy = _FakeStrategy(_turn())
    service = _service(
        strategy, agent=_researcher_agent(tools=["get_current_time"]), tools=_tool_registry()
    )

    await service.run("hi", "researcher", tools=[])

    assert strategy.last_tools == {}


async def test_run_given_request_tools_overrides_agent_tools():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(tools=[]), tools=_tool_registry())

    await service.run("hi", "researcher", tools=["get_current_time"])

    assert strategy.last_tools is not None
    assert list(strategy.last_tools) == ["get_current_time"]


async def test_run_given_duplicate_request_tools_resolves_to_one_tool():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(tools=[]), tools=_tool_registry())

    await service.run("hi", "researcher", tools=["get_current_time", "get_current_time"])

    assert strategy.last_tools is not None
    assert list(strategy.last_tools) == ["get_current_time"]


async def test_run_given_unregistered_tool_raises_tool_not_found_error():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(tools=["missing"]))

    with pytest.raises(ToolNotFoundError):
        await service.run("hi", "researcher")


async def test_run_passes_agent_max_tool_iterations_to_strategy():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(max_tool_iterations=3))

    await service.run("hi", "researcher")

    assert strategy.last_max_iterations == 3


async def test_run_passes_agent_max_tool_result_chars_to_strategy():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(max_tool_result_chars=500))

    await service.run("hi", "researcher")

    assert strategy.last_max_tool_result_chars == 500


async def test_run_given_no_max_tool_result_chars_passes_none_to_strategy():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent())

    await service.run("hi", "researcher")

    assert strategy.last_max_tool_result_chars is None


async def test_run_given_context_window_exceeded_retry_passes_max_tool_result_chars_too():
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    await session_store.append("researcher", session_id, [Message(role="user", content="hi")])
    compaction_service = _FakeCompactionService(compact_result=True, session_store=session_store)
    strategy = _FakeStrategy([LLMContextWindowExceededError("too big"), _turn("recovered")])
    service = _service(
        strategy,
        agent=_researcher_agent(max_tool_result_chars=500),
        session_store=session_store,
        compaction_service=compaction_service,
    )

    await service.run("again", "researcher", session_id=session_id)

    assert strategy.last_max_tool_result_chars == 500


async def test_run_builds_run_from_strategys_turn():
    turn = Turn(
        messages=[Message(role="assistant", content="the answer")],
        usage=Usage(prompt_tokens=7, completion_tokens=4, total_tokens=11),
        finish_reason="stop",
        final_total_tokens=11,
    )
    service = _service(_FakeStrategy(turn))

    run = await service.run("hi", "researcher")

    assert run.response == turn.message
    assert run.usage == turn.usage
    assert run.finish_reason == "stop"


async def test_run_given_no_session_id_creates_a_new_session_and_returns_its_id():
    session_store = InMemorySessionStore()
    service = _service(_FakeStrategy(_turn()), session_store=session_store)

    run = await service.run("hi", "researcher")

    assert run.session_id != ""
    stored = await session_store.get("researcher", run.session_id)
    assert stored[0] == Message(role="user", content="hi")
    assert stored[-1] == Message(role="assistant", content="hi there")


async def test_run_given_no_session_id_stores_only_user_and_turn_messages_not_system():
    session_store = InMemorySessionStore()
    service = _service(_FakeStrategy(_turn()), session_store=session_store)

    run = await service.run("hi", "researcher")

    stored = await session_store.get("researcher", run.session_id)
    assert Message(role="system", content="You are a research assistant.") not in stored


async def test_run_given_existing_session_id_threads_stored_history_into_messages():
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    await session_store.append(
        "researcher",
        session_id,
        [Message(role="user", content="what's 2+2?"), Message(role="assistant", content="4")],
    )
    strategy = _FakeStrategy(_turn("noted"))
    service = _service(strategy, session_store=session_store)

    await service.run("thanks", "researcher", session_id=session_id)

    assert strategy.last_messages == [
        Message(role="system", content="You are a research assistant."),
        Message(role="user", content="what's 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="thanks"),
    ]


async def test_run_given_existing_session_id_appends_new_turn_onto_existing_history():
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    await session_store.append("researcher", session_id, [Message(role="user", content="hi")])
    service = _service(_FakeStrategy(_turn("hi there")), session_store=session_store)

    run = await service.run("again", "researcher", session_id=session_id)

    assert run.session_id == session_id
    stored = await session_store.get("researcher", session_id)
    assert stored == [
        Message(role="user", content="hi"),
        Message(role="user", content="again"),
        Message(role="assistant", content="hi there"),
    ]


async def test_run_given_unknown_session_id_raises_session_not_found_error():
    service = _service(_FakeStrategy(_turn()))

    with pytest.raises(SessionNotFoundError):
        await service.run("hi", "researcher", session_id="does-not-exist")


async def test_run_given_max_input_chars_unset_never_raises_regardless_of_length():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent())

    await service.run("x" * 1_000_000, "researcher")

    assert strategy.last_messages is not None


async def test_run_given_message_within_max_input_chars_succeeds():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(max_input_chars=10))

    await service.run("short", "researcher")

    assert strategy.last_messages is not None


async def test_run_given_message_exceeds_max_input_chars_raises_input_too_large_error():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent(max_input_chars=5))

    with pytest.raises(InputTooLargeError):
        await service.run("this is too long", "researcher")

    assert strategy.last_messages is None


async def test_run_given_no_compaction_service_never_touches_it_and_behaves_as_before():
    strategy = _FakeStrategy(_turn())
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    service = _service(strategy, session_store=session_store, compaction_service=None)

    run = await service.run("hi", "researcher", session_id=session_id)

    assert run.session_id == session_id


async def test_run_given_existing_session_id_calls_maybe_compact_before_building_messages():
    compaction_service = _FakeCompactionService()
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, session_store=session_store, compaction_service=compaction_service)

    await service.run("hi", "researcher", session_id=session_id)

    assert compaction_service.maybe_compact_calls == [("researcher", session_id, "openai/gpt-4o")]


async def test_run_given_no_session_id_never_calls_maybe_compact():
    compaction_service = _FakeCompactionService()
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, compaction_service=compaction_service)

    await service.run("hi", "researcher")

    assert compaction_service.maybe_compact_calls == []


async def test_run_given_compaction_service_records_usage_after_successful_append():
    compaction_service = _FakeCompactionService()
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, session_store=session_store, compaction_service=compaction_service)

    run = await service.run("hi", "researcher", session_id=session_id)

    assert compaction_service.record_usage_calls == [("researcher", run.session_id, 5)]


async def test_run_given_context_window_exceeded_compacts_and_retries_once():
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    await session_store.append("researcher", session_id, [Message(role="user", content="hi")])
    compaction_service = _FakeCompactionService(compact_result=True, session_store=session_store)
    strategy = _FakeStrategy([LLMContextWindowExceededError("too big"), _turn("recovered")])
    service = _service(strategy, session_store=session_store, compaction_service=compaction_service)

    run = await service.run("again", "researcher", session_id=session_id)

    assert compaction_service.compact_calls == [("researcher", session_id)]
    assert run.response.content == "recovered"
    # The retry must rebuild its messages from the freshly-compacted history, not reuse
    # the stale pre-compaction list from the first (failed) call.
    assert strategy.call_messages[0] != strategy.call_messages[1]


async def test_run_given_retry_also_overflows_raises_compaction_exhausted_error():
    # keep_recent_turns is never overridden -- one retry only, no escalation to a more
    # aggressive compaction pass. If that single retry still overflows, the request fails.
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    await session_store.append("researcher", session_id, [Message(role="user", content="hi")])
    compaction_service = _FakeCompactionService(compact_result=True, session_store=session_store)
    strategy = _FakeStrategy(
        [
            LLMContextWindowExceededError("too big"),
            LLMContextWindowExceededError("still too big"),
        ]
    )
    service = _service(strategy, session_store=session_store, compaction_service=compaction_service)

    with pytest.raises(CompactionExhaustedError):
        await service.run("again", "researcher", session_id=session_id)

    assert compaction_service.compact_calls == [("researcher", session_id)]


async def test_run_given_compact_returns_false_raises_compaction_exhausted_without_retrying():
    compaction_service = _FakeCompactionService(compact_result=False)
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    strategy = _FakeStrategy([LLMContextWindowExceededError("too big"), _turn("unreached")])
    service = _service(strategy, session_store=session_store, compaction_service=compaction_service)

    with pytest.raises(CompactionExhaustedError):
        await service.run("hi", "researcher", session_id=session_id)

    assert compaction_service.compact_calls == [("researcher", session_id)]
    assert strategy._call_count == 1


async def test_run_given_no_compaction_service_context_window_exceeded_propagates_plain_error():
    # Compaction was never available to try, so this must stay the generic overflow error --
    # `CompactionExhaustedError` means specifically "tried everything and it didn't help".
    strategy = _FakeStrategy([LLMContextWindowExceededError("too big"), _turn("unreached")])
    session_store = InMemorySessionStore()
    session_id = await session_store.create("researcher")
    service = _service(strategy, session_store=session_store, compaction_service=None)

    with pytest.raises(LLMContextWindowExceededError) as exc_info:
        await service.run("hi", "researcher", session_id=session_id)

    assert not isinstance(exc_info.value, CompactionExhaustedError)


async def test_run_given_fresh_session_context_window_exceeded_never_retries():
    compaction_service = _FakeCompactionService(compact_result=True)
    strategy = _FakeStrategy([LLMContextWindowExceededError("too big"), _turn("unreached")])
    service = _service(strategy, compaction_service=compaction_service)

    with pytest.raises(LLMContextWindowExceededError) as exc_info:
        await service.run("hi", "researcher")

    assert not isinstance(exc_info.value, CompactionExhaustedError)
    assert compaction_service.compact_calls == []


async def test_run_passes_agent_tool_call_and_total_char_caps_to_strategy():
    strategy = _FakeStrategy(_turn())
    service = _service(
        strategy,
        agent=_researcher_agent(max_tool_calls_per_round=3, max_tool_results_total_chars=1000),
    )

    await service.run("hi", "researcher")

    assert (strategy.last_max_tool_calls_per_round, strategy.last_max_tool_results_total_chars) == (
        3,
        1000,
    )


async def test_run_given_no_tool_call_and_total_char_caps_passes_none_to_strategy():
    strategy = _FakeStrategy(_turn())
    service = _service(strategy, agent=_researcher_agent())

    await service.run("hi", "researcher")

    assert (strategy.last_max_tool_calls_per_round, strategy.last_max_tool_results_total_chars) == (
        None,
        None,
    )
