"""Agent run orchestration."""

from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.protocols.isession_store import ISessionStore
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`."""
    return a if a is not None else b


class AgentRunService:
    """Orchestrates a single agent run: resolves config, delegates reasoning to a strategy.

    Always routed through a registered agent's system prompt, defaults, and tools.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        strategy_registry: StrategyRegistry,
        base_prompt: str | None,
        session_store: ISessionStore,
    ) -> None:
        """Initialize AgentRunService with its registries.

        Args:
            llm_registry: Registry of available LLM implementations.
            agent_registry: Registry of available agent configurations.
            tool_registry: Registry of available tool implementations.
            strategy_registry: Registry of available reasoning strategies.
            base_prompt: Prepended before every agent's `system_prompt`, merged into
                the same leading system message. `None` if there's nothing to share.
            session_store: Per-conversation message history storage.
        """
        self._llm_registry = llm_registry
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._strategy_registry = strategy_registry
        self._base_prompt = base_prompt
        self._session_store = session_store

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

        `model`/`strategy`, if given, override the agent's configured `model`/`strategy`.
        `temperature`/`top_p`/`max_tokens` resolve independently as: this call's value,
        else the agent's configured value, else the LLM's own configured default. The
        agent's `system_prompt` is unconditionally prepended as the leading system message.

        `tools` resolves as a tri-state: omitted or `None` uses the agent's configured
        `tools` (or none, if it has none); an explicit empty list suppresses tools
        entirely; a non-empty list is used exactly as given. Each resolved name is looked
        up against `ToolRegistry` here, before the strategy ever runs, so a strategy only
        ever sees the exact tool instances it was given -- never the registry itself.

        `session_id`, if given, must belong to `agent` (a session is locked to its
        creating agent) -- its stored history is threaded in between the system message
        and this call's new user message. If omitted, the conversation starts empty and a
        new session is created to hold it. Either way, this call's user message and
        everything the strategy generates are appended to the session afterward -- the
        system message itself is never stored, since it's rebuilt fresh from the current
        `agent_config`/`base_prompt` every call.

        Raises:
            AgentNotFoundError: if `agent` is not registered.
            LLMNotFoundError: if the resolved model is not registered.
            StrategyNotFoundError: if the resolved strategy is not registered.
            ToolNotFoundError: if a resolved tool name is not registered.
            SessionNotFoundError: if `session_id` is given but no session exists for it
                under this agent.
            LLMError: if the underlying LLM call fails.
        """
        agent_config = self._agent_registry.get(agent)
        system_content = agent_config.system_prompt
        if self._base_prompt is not None:
            system_content = f"{self._base_prompt}\n\n{agent_config.system_prompt}"

        history = [] if session_id is None else await self._session_store.get(agent, session_id)
        user_message = Message(role="user", content=message)
        messages = [Message(role="system", content=system_content), *history, user_message]

        effective_model = model if model is not None else agent_config.model
        effective_strategy = strategy if strategy is not None else agent_config.strategy

        resolved_temperature = _first_not_none(temperature, agent_config.temperature)
        resolved_top_p = _first_not_none(top_p, agent_config.top_p)
        resolved_max_tokens = _first_not_none(max_tokens, agent_config.max_tokens)

        tool_names = tools if tools is not None else agent_config.tools
        resolved_tools = {name: self._tool_registry.get(name) for name in tool_names}

        llm = self._llm_registry.get(effective_model)
        strategy_instance = self._strategy_registry.get(effective_strategy)
        turn = await strategy_instance.run(
            messages,
            llm,
            resolved_tools,
            agent_config.max_tool_iterations,
            temperature=resolved_temperature,
            top_p=resolved_top_p,
            max_tokens=resolved_max_tokens,
        )

        if session_id is None:
            session_id = await self._session_store.create(agent)
        await self._session_store.append(agent, session_id, [user_message, *turn.messages])

        return Run(
            model=effective_model,
            response=turn.message,
            usage=turn.usage,
            finish_reason=turn.finish_reason,
            session_id=session_id,
        )
