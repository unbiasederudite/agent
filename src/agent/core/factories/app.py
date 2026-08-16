"""Factory for building runtime registries from configuration."""

import logging
from pathlib import Path

from pydantic import ValidationError

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import ConfigError
from agent.core.models.config import AppConfig
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry


def build_registries(config_path: Path) -> tuple[LLMRegistry, AgentRegistry]:
    """Load AppConfig from `config_path` and return populated LLMRegistry and AgentRegistry.

    Raises:
        ConfigError: if the file is missing, not valid JSON, fails AppConfig validation,
            declares a duplicate LLM model or agent name, or an agent's `default_llm`
            doesn't match any configured LLM.
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
        agent_registry.register(agent_config.name, agent_config)

    return llm_registry, agent_registry
