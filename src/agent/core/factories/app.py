"""Factory for building runtime registries from configuration."""

from collections.abc import Callable

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import ConfigError
from agent.core.models.config import (
    AgentConfig,
    AppConfig,
    CompactionConfig,
    LLMConfig,
    LoggingConfig,
    StrategyConfig,
    ToolConfig,
)
from agent.core.protocols.istrategy import IStrategy
from agent.core.protocols.itool import ITool
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.strategies.react import ReactStrategy
from agent.core.tools.get_current_time import GetCurrentTimeTool

_TOOL_IMPLEMENTATIONS: dict[str, Callable[[], ITool]] = {
    "get_current_time": GetCurrentTimeTool,
}

_STRATEGY_IMPLEMENTATIONS: dict[str, Callable[[], IStrategy]] = {
    "react": ReactStrategy,
}


def _build_llm_registry(llm_configs: list[LLMConfig]) -> tuple[LLMRegistry, set[str]]:
    """Build the LLM registry, raising ConfigError on a duplicate model name."""
    llm_registry = LLMRegistry()
    seen_models: set[str] = set()
    for llm_config in llm_configs:
        if llm_config.model in seen_models:
            raise ConfigError(f"duplicate LLM model in config: {llm_config.model}")
        seen_models.add(llm_config.model)
        llm_registry.register(llm_config.model, LiteLLMAdapter.from_config(llm_config))
    return llm_registry, seen_models


def _build_tool_registry(tool_configs: list[ToolConfig]) -> tuple[ToolRegistry, set[str]]:
    """Build the tool registry, raising ConfigError on a duplicate name or missing impl."""
    tool_registry = ToolRegistry()
    seen_tools: set[str] = set()
    for tool_config in tool_configs:
        if tool_config.name in seen_tools:
            raise ConfigError(f"duplicate tool name in config: {tool_config.name}")
        seen_tools.add(tool_config.name)
        implementation = _TOOL_IMPLEMENTATIONS.get(tool_config.name)
        if implementation is None:
            raise ConfigError(f"no tool implementation registered for: {tool_config.name}")
        tool_registry.register(tool_config.name, implementation())
    return tool_registry, seen_tools


def _build_strategy_registry(
    strategy_configs: list[StrategyConfig],
) -> tuple[StrategyRegistry, set[str]]:
    """Build the strategy registry, raising ConfigError on a duplicate name or missing impl."""
    strategy_registry = StrategyRegistry()
    seen_strategies: set[str] = set()
    for strategy_config in strategy_configs:
        if strategy_config.name in seen_strategies:
            raise ConfigError(f"duplicate strategy name in config: {strategy_config.name}")
        seen_strategies.add(strategy_config.name)
        implementation = _STRATEGY_IMPLEMENTATIONS.get(strategy_config.name)
        if implementation is None:
            raise ConfigError(f"no strategy implementation registered for: {strategy_config.name}")
        strategy_registry.register(strategy_config.name, implementation())
    return strategy_registry, seen_strategies


def _build_agent_registry(
    agent_configs: list[AgentConfig],
    known_models: set[str],
    known_tools: set[str],
    known_strategies: set[str],
) -> AgentRegistry:
    """Build the agent registry, raising ConfigError on any invalid agent declaration."""
    agent_registry = AgentRegistry()
    seen_agents: set[str] = set()
    for agent_config in agent_configs:
        if agent_config.name in seen_agents:
            raise ConfigError(f"duplicate agent name in config: {agent_config.name}")
        seen_agents.add(agent_config.name)
        if agent_config.model not in known_models:
            raise ConfigError(
                f"agent '{agent_config.name}' declares unknown model: {agent_config.model}"
            )
        if agent_config.strategy not in known_strategies:
            raise ConfigError(
                f"agent '{agent_config.name}' declares unknown strategy: {agent_config.strategy}"
            )
        seen_agent_tools: set[str] = set()
        for tool_name in agent_config.tools:
            if tool_name not in known_tools:
                raise ConfigError(f"agent '{agent_config.name}' declares unknown tool: {tool_name}")
            if tool_name in seen_agent_tools:
                raise ConfigError(
                    f"agent '{agent_config.name}' declares duplicate tool: {tool_name}"
                )
            seen_agent_tools.add(tool_name)
        agent_registry.register(agent_config.name, agent_config)
    return agent_registry


def build_registries(
    config: AppConfig,
) -> tuple[
    LLMRegistry,
    AgentRegistry,
    ToolRegistry,
    StrategyRegistry,
    str | None,
    CompactionConfig | None,
    LoggingConfig,
    int | None,
]:
    """Build populated registries plus raw config from an already-loaded `AppConfig`.

    Takes a parsed `AppConfig`, not a file path -- reading/parsing the config file is I/O,
    which this factory (like the rest of `core/`) owns none of; the caller (`api/app.py`'s
    `create_app()`) reads and validates the JSON itself and passes the result in here.

    Returns the registries, `base_prompt`, and the raw `compaction` config. `create_app()`
    builds `CompactionService` from the raw config, since that also needs `session_store`,
    a process-level object with no config surface of its own. Also returns the raw `logging`
    config -- `create_app()` builds its own logging setup from it, since that also needs
    things (a request-id filter, whether this is even an HTTP context) not owned by this
    factory, the same reasoning `compaction_config` already documents here. Also returns the
    raw `max_sessions` value -- `create_app()` passes it straight to `InMemorySessionStore`'s
    constructor, since this factory has no session-store object of its own to configure.

    Raises:
        ConfigError: if `config` declares a duplicate LLM model, agent name, tool name, or
            strategy name, an agent's `model` doesn't match any configured LLM, an agent's
            `strategy` doesn't match any configured strategy, an agent's `tools` entry
            doesn't match any configured tool or repeats a tool name, a configured tool or
            strategy name has no matching code-level implementation, `compaction.model`
            doesn't match any configured LLM, or an agent's `allowed_tools`,
            `allowed_models`, or `allowed_strategies` entry doesn't match any configured
            tool, model, or strategy.
    """
    llm_registry, known_models = _build_llm_registry(config.llms)
    tool_registry, known_tools = _build_tool_registry(config.tools)
    strategy_registry, known_strategies = _build_strategy_registry(config.strategies)
    agent_registry = _build_agent_registry(
        config.agents, known_models, known_tools, known_strategies
    )
    if config.compaction is not None and config.compaction.model not in known_models:
        raise ConfigError(f"compaction declares unknown model: {config.compaction.model}")

    for agent_config in config.agents:
        if agent_config.allowed_tools is not None:
            for tool_name in agent_config.allowed_tools:
                if tool_name not in known_tools:
                    raise ConfigError(
                        f"agent '{agent_config.name}' declares unknown tool in "
                        f"allowed_tools: {tool_name}"
                    )
        if agent_config.allowed_models is not None:
            for model_name in agent_config.allowed_models:
                if model_name not in known_models:
                    raise ConfigError(
                        f"agent '{agent_config.name}' declares unknown model in "
                        f"allowed_models: {model_name}"
                    )
        if agent_config.allowed_strategies is not None:
            for strategy_name in agent_config.allowed_strategies:
                if strategy_name not in known_strategies:
                    raise ConfigError(
                        f"agent '{agent_config.name}' declares unknown strategy in "
                        f"allowed_strategies: {strategy_name}"
                    )

    return (
        llm_registry,
        agent_registry,
        tool_registry,
        strategy_registry,
        config.base_prompt,
        config.compaction,
        config.logging,
        config.max_sessions,
    )
