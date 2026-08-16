"""Factory for building runtime registries from configuration."""

import logging
from pathlib import Path

from pydantic import ValidationError

from agent.adapters.litellm import LiteLLMAdapter
from agent.core.exceptions import ConfigError
from agent.core.models.config import AppConfig
from agent.core.registries.llm import LLMRegistry


def build_llm_registry(config_path: Path) -> LLMRegistry:
    """Load AppConfig from `config_path` and return a populated LLMRegistry.

    Raises:
        ConfigError: if the file is missing, not valid JSON, fails AppConfig
            validation, or declares a duplicate LLM model name.
    """
    try:
        config = AppConfig.model_validate_json(config_path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ConfigError(str(exc)) from exc

    logging.basicConfig(level=config.logging.level, force=True)

    registry = LLMRegistry()
    seen: set[str] = set()
    for llm_config in config.llms:
        if llm_config.model in seen:
            raise ConfigError(f"duplicate LLM model in config: {llm_config.model}")
        seen.add(llm_config.model)
        registry.register(llm_config.model, LiteLLMAdapter(llm_config.model))
    return registry
