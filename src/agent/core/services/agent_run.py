"""Agent run orchestration."""

import asyncio
import logging
import time
from contextlib import AbstractAsyncContextManager, nullcontext

from agent.core.exceptions import (
    CompactionExhaustedError,
    InputTooLargeError,
    LLMContextWindowExceededError,
    ModelNotAllowedError,
    RequestTimeoutError,
    StrategyNotAllowedError,
    ToolNotAllowedError,
)
from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.models.turn import Turn
from agent.core.protocols.isession_store import ISessionStore
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.run_context import run_context, update_session_id
from agent.core.services.compaction import CompactionService
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.services.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`.

    Args:
        a: Preferred value.
        b: Fallback value.

    Returns:
        T | None: `a` if not `None`, else `b`.
    """
    return a if a is not None else b


class AgentRunService:
    """Orchestrates a single agent run: resolves config, delegates reasoning to a strategy."""

    def __init__(
        self,
        llm_registry: LLMRegistry,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        strategy_registry: StrategyRegistry,
        base_prompt: str | None,
        session_store: ISessionStore,
        compaction_service: CompactionService | None = None,
        cost_tracker: CostTracker | None = None,
        context_tracker: ContextFootprintTracker | None = None,
    ) -> None:
        """Initialize with its registries and dependencies.

        Args:
            llm_registry: Registry of available LLM implementations.
            agent_registry: Registry of available agent configurations.
            tool_registry: Registry of available tool implementations.
            strategy_registry: Registry of available reasoning strategies.
            base_prompt: Text prepended before every agent's system prompt.
            session_store: Per-conversation message history storage.
            compaction_service: Keeps a session's history under a token budget.
            cost_tracker: Cumulative per-session/per-agent token and cost usage.
            context_tracker: Each session's current context-token footprint.
        """
        self._llm_registry = llm_registry
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._strategy_registry = strategy_registry
        self._base_prompt = base_prompt
        self._session_store = session_store
        self._compaction_service = compaction_service
        self._cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self._context_tracker = (
            context_tracker if context_tracker is not None else ContextFootprintTracker()
        )

    async def run(
        self,
        message: str,
        agent: str,
        *,
        model: str | None = None,
        strategy: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
        session_id: str | None = None,
    ) -> Run:
        """Complete `message`, routed through the registered agent `agent`.

        Args:
            message: The user's message to send.
            agent: Name of the agent to run.
            model: Model override.
            strategy: Reasoning strategy override.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling override.
            max_tokens: Max output tokens override.
            tools: Tool names to offer the LLM, overriding the agent's configured tools.
            session_id: Session to continue. Omit to start a new one.

        Returns:
            Run: the completed run.

        Raises:
            AgentNotFoundError: `agent` is not registered.
            InputTooLargeError: `message` exceeds the agent's `max_input_chars`.
            LLMNotFoundError: the resolved model is not registered.
            StrategyNotFoundError: the resolved strategy is not registered.
            ToolNotFoundError: a resolved tool name is not registered.
            ToolNotAllowedError: a resolved tool isn't in the agent's allowed tools.
            ModelNotAllowedError: the resolved model isn't in the agent's allowed models.
            StrategyNotAllowedError: the resolved strategy isn't in the agent's allowed
                strategies.
            SessionNotFoundError: no session exists for `session_id` under this agent.
            LLMContextWindowExceededError: the call overflowed the model's context
                window with no compaction available.
            CompactionExhaustedError: the call overflowed and the compaction retry was
                exhausted.
            LLMError: the underlying LLM call failed.
            SessionBusyError: another operation is already using `session_id`.
            RequestTimeoutError: the call exceeded the agent's `max_request_seconds`.
        """
        busy_guard: AbstractAsyncContextManager[None] = (
            self._session_store.busy(agent, session_id) if session_id is not None else nullcontext()
        )
        with run_context(agent, session_id):
            async with busy_guard:
                return await self._run_body(
                    message,
                    agent,
                    model=model,
                    strategy=strategy,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    tools=tools,
                    session_id=session_id,
                )

    async def _run_body(
        self,
        message: str,
        agent: str,
        *,
        model: str | None,
        strategy: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[str] | None,
        session_id: str | None,
    ) -> Run:
        """Resolve config, build messages, and execute the strategy call for one run.

        Args:
            message: The user's message to send.
            agent: Name of the agent to run.
            model: Model override.
            strategy: Reasoning strategy override.
            temperature: Sampling temperature override.
            top_p: Nucleus sampling override.
            max_tokens: Max output tokens override.
            tools: Tool names to offer the LLM, overriding the agent's configured tools.
            session_id: Session to continue, or `None` for a new one.

        Returns:
            Run: the completed run.

        Raises:
            AgentNotFoundError: `agent` is not registered.
            InputTooLargeError: `message` exceeds the agent's `max_input_chars`.
            LLMNotFoundError: the resolved model is not registered.
            StrategyNotFoundError: the resolved strategy is not registered.
            ToolNotFoundError: a resolved tool name is not registered.
            ToolNotAllowedError: a resolved tool isn't in the agent's allowed tools.
            ModelNotAllowedError: the resolved model isn't in the agent's allowed models.
            StrategyNotAllowedError: the resolved strategy isn't in the agent's allowed
                strategies.
            SessionNotFoundError: no session exists for `session_id` under this agent.
            LLMContextWindowExceededError: the call overflowed the model's context
                window with no compaction available.
            CompactionExhaustedError: the call overflowed and the compaction retry was
                exhausted.
            LLMError: the underlying LLM call failed.
            RequestTimeoutError: the call exceeded the agent's `max_request_seconds`.
        """
        start = time.monotonic()
        agent_config = self._agent_registry.get(agent)
        if agent_config.max_input_chars is not None and len(message) > agent_config.max_input_chars:
            logger.info(
                "message length %d exceeds agent '%s's max_input_chars (%d)",
                len(message),
                agent,
                agent_config.max_input_chars,
            )
            raise InputTooLargeError(
                f"message length {len(message)} exceeds agent '{agent}'s max_input_chars "
                f"({agent_config.max_input_chars})"
            )

        system_content = agent_config.system_prompt
        if self._base_prompt is not None:
            system_content = f"{self._base_prompt}\n\n{agent_config.system_prompt}"

        effective_model = model if model is not None else agent_config.model
        self._llm_registry.get(effective_model)
        if (
            agent_config.allowed_models is not None
            and effective_model not in agent_config.allowed_models
        ):
            logger.info("model '%s' is not allowed for agent '%s'", effective_model, agent)
            raise ModelNotAllowedError(
                f"model '{effective_model}' is not allowed for agent '{agent}'"
            )
        effective_strategy = strategy if strategy is not None else agent_config.strategy
        self._strategy_registry.get(effective_strategy)
        if (
            agent_config.allowed_strategies is not None
            and effective_strategy not in agent_config.allowed_strategies
        ):
            logger.info("strategy '%s' is not allowed for agent '%s'", effective_strategy, agent)
            raise StrategyNotAllowedError(
                f"strategy '{effective_strategy}' is not allowed for agent '{agent}'"
            )

        tool_names = tools if tools is not None else agent_config.tools
        resolved_tools = {name: self._tool_registry.get(name) for name in tool_names}
        if agent_config.allowed_tools is not None:
            disallowed = [name for name in tool_names if name not in agent_config.allowed_tools]
            if disallowed:
                logger.info("tools %s are not allowed for agent '%s'", disallowed, agent)
                raise ToolNotAllowedError(f"tools {disallowed} are not allowed for agent '{agent}'")

        logger.info(
            "agent run started: agent=%s session=%s model=%s strategy=%s",
            agent,
            session_id if session_id is not None else "new",
            effective_model,
            effective_strategy,
        )

        user_message = Message(role="user", content=message)

        async def _run_within_deadline() -> Turn:
            if session_id is not None and self._compaction_service is not None:
                await self._compaction_service.maybe_compact(agent, session_id, effective_model)

            history = [] if session_id is None else await self._session_store.get(agent, session_id)
            messages = [Message(role="system", content=system_content), *history, user_message]

            resolved_temperature = _first_not_none(temperature, agent_config.temperature)
            resolved_top_p = _first_not_none(top_p, agent_config.top_p)
            resolved_max_tokens = _first_not_none(max_tokens, agent_config.max_tokens)

            llm = self._llm_registry.get(effective_model)
            strategy_instance = self._strategy_registry.get(effective_strategy)

            async def _call_strategy(call_messages: list[Message]) -> Turn:
                return await strategy_instance.run(
                    call_messages,
                    llm,
                    resolved_tools,
                    agent_config.max_tool_iterations,
                    temperature=resolved_temperature,
                    top_p=resolved_top_p,
                    max_tokens=resolved_max_tokens,
                    max_tool_result_chars=agent_config.max_tool_result_chars,
                    max_tool_calls_per_round=agent_config.max_tool_calls_per_round,
                    max_tool_results_total_chars=agent_config.max_tool_results_total_chars,
                )

            try:
                return await _call_strategy(messages)
            except LLMContextWindowExceededError as exc:
                logger.warning(
                    "context window overflow for agent=%s session=%s despite the proactive check",
                    agent,
                    session_id,
                )
                if self._compaction_service is None or session_id is None:
                    raise
                retried = False
                result: Turn | None = None
                if await self._compaction_service.compact(agent, session_id):
                    history = await self._session_store.get(agent, session_id)
                    messages = [
                        Message(role="system", content=system_content),
                        *history,
                        user_message,
                    ]
                    try:
                        result = await _call_strategy(messages)
                        retried = True
                    except LLMContextWindowExceededError:
                        pass
                if retried:
                    logger.info(
                        "reactive compact-and-retry succeeded for agent=%s session=%s",
                        agent,
                        session_id,
                    )
                    assert result is not None
                    return result
                logger.error(
                    "reactive compact-and-retry exhausted for agent=%s session=%s, "
                    "raising CompactionExhaustedError",
                    agent,
                    session_id,
                    exc_info=True,
                    extra={"exception_type": "CompactionExhaustedError"},
                )
                raise CompactionExhaustedError(str(exc)) from exc

        try:
            if agent_config.max_request_seconds is not None:
                # A timeout here can fire after a proactive compaction already committed
                # its rewrite to the session store; that rewrite is not rolled back.
                turn = await asyncio.wait_for(
                    _run_within_deadline(), timeout=agent_config.max_request_seconds
                )
            else:
                turn = await _run_within_deadline()
        except TimeoutError as exc:
            logger.warning(
                "agent run exceeded max_request_seconds=%.1fs for agent=%s session=%s",
                agent_config.max_request_seconds,
                agent,
                session_id,
            )
            raise RequestTimeoutError(
                f"agent '{agent}' exceeded its {agent_config.max_request_seconds}s request budget"
            ) from exc

        if session_id is None:
            session_id = await self._session_store.create(agent)
            update_session_id(session_id)
        async with self._session_store.lock(agent, session_id):
            await self._session_store.append(agent, session_id, [user_message, *turn.messages])
        self._cost_tracker.record(agent, session_id, turn.usage)
        self._context_tracker.record(agent, session_id, turn.final_total_tokens)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "agent run completed: agent=%s session=%s finish_reason=%s total_tokens=%d, %.1fms",
            agent,
            session_id,
            turn.finish_reason,
            turn.usage.total_tokens,
            duration_ms,
        )

        return Run(
            model=effective_model,
            response=turn.message,
            usage=turn.usage,
            finish_reason=turn.finish_reason,
            session_id=session_id,
        )
