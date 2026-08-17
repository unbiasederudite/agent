"""Factory for building runtime registries from configuration."""

import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import ConfigError
from agent.core.models.config import AppConfig
from agent.core.protocols.itool import ITool
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.tools.get_current_time import GetCurrentTimeTool

_TOOL_IMPLEMENTATIONS: dict[str, Callable[[], ITool]] = {
    "get_current_time": GetCurrentTimeTool,
}


def build_registries(config_path: Path) -> tuple[LLMRegistry, AgentRegistry, ToolRegistry]:
    """Load AppConfig from `config_path` and return populated registries.

    Raises:
        ConfigError: if the file is missing, not valid JSON, fails AppConfig validation,
            declares a duplicate LLM model, agent name, or tool name, an agent's
            `default_llm` doesn't match any configured LLM, an agent's `tools` entry
            doesn't match any configured tool or repeats a tool name, or a configured
            tool name has no matching code-level implementation.
    """
    try:
        config = AppConfig.model_validate_json(config_path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ConfigError(str(exc)) from exc

    logging.basicConfig(level=config.logging.level, force=True)

    llm_registry = LLMRegistry()
    seen_models: set[str] = set()
    for llm_config in config.llms:
        if llm_config.model in seen_models:
            raise ConfigError(f"duplicate LLM model in config: {llm_config.model}")
        seen_models.add(llm_config.model)
        llm_registry.register(
            llm_config.model,
            LiteLLMAdapter(
                llm_config.model,
                temperature=llm_config.temperature,
                top_p=llm_config.top_p,
                max_tokens=llm_config.max_tokens,
            ),
        )

    tool_registry = ToolRegistry()
    seen_tools: set[str] = set()
    for tool_config in config.tools:
        if tool_config.name in seen_tools:
            raise ConfigError(f"duplicate tool name in config: {tool_config.name}")
        seen_tools.add(tool_config.name)
        implementation = _TOOL_IMPLEMENTATIONS.get(tool_config.name)
        if implementation is None:
            raise ConfigError(f"no tool implementation registered for: {tool_config.name}")
        tool_registry.register(tool_config.name, implementation())

    agent_registry = AgentRegistry()
    seen_agents: set[str] = set()
    for agent_config in config.agents:
        if agent_config.name in seen_agents:
            raise ConfigError(f"duplicate agent name in config: {agent_config.name}")
        seen_agents.add(agent_config.name)
        if agent_config.default_llm not in seen_models:
            raise ConfigError(
                f"agent '{agent_config.name}' declares unknown default_llm: "
                f"{agent_config.default_llm}"
            )
        seen_agent_tools: set[str] = set()
        for tool_name in agent_config.tools:
            if tool_name not in seen_tools:
                raise ConfigError(f"agent '{agent_config.name}' declares unknown tool: {tool_name}")
            if tool_name in seen_agent_tools:
                raise ConfigError(
                    f"agent '{agent_config.name}' declares duplicate tool: {tool_name}"
                )
            seen_agent_tools.add(tool_name)
        agent_registry.register(agent_config.name, agent_config)

    return llm_registry, agent_registry, tool_registry
