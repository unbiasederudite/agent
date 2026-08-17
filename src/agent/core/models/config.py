"""Pydantic config models for application startup."""

from typing import Literal

from pydantic import BaseModel, Field


class SamplingDefaults(BaseModel):
    """Sampling-default fields shared by `LLMConfig` and `AgentConfig`."""

    temperature: float | None = Field(
        default=None, description="Default sampling temperature, if set."
    )
    top_p: float | None = Field(default=None, description="Default nucleus sampling value, if set.")
    max_tokens: int | None = Field(default=None, description="Default max output tokens, if set.")


class LLMConfig(SamplingDefaults):
    """One entry in the startup LLM allow-list.

    `model` is a litellm-format provider/model id (e.g. "anthropic/claude-sonnet-5")
    and doubles as the LLMRegistry lookup key.
    """

    model: str = Field(
        description='litellm-format provider/model id, e.g. "anthropic/claude-sonnet-5".'
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level to emit."
    )


class ToolConfig(BaseModel):
    """One entry in the startup tool allow-list.

    `name` must match a code-level tool implementation (see `core/factories/app.py`)
    and is also the ToolRegistry lookup key.
    """

    name: str = Field(description="The tool's lookup key, matching a code-level implementation.")


class AgentConfig(SamplingDefaults):
    """One entry in the startup agent allow-list.

    `name` is the AgentRegistry lookup key. `system_prompt` is unconditionally prepended
    as the leading system message whenever this agent is selected -- it is identity
    content, not a request-overridable value. `default_llm` must match a declared
    `LLMConfig.model` in the same config. Every name in `tools` must match a declared
    `ToolConfig.name` in the same config.
    """

    name: str = Field(description="The agent's lookup key in the AgentRegistry.")
    system_prompt: str = Field(
        description=(
            "Unconditionally prepended as the leading system message when this agent "
            "is selected; never overridden by client-supplied messages."
        )
    )
    default_llm: str = Field(
        description="The LLMConfig.model used when the request doesn't override `model`."
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names available to this agent by default, unless the request "
            "overrides them (see CompletionService.run's tri-state `tools` resolution)."
        ),
    )


class AppConfig(BaseModel):
    """Root startup configuration, loaded once from a JSON file."""

    llms: list[LLMConfig] = Field(description="The allow-list of LLMs available to this process.")
    agents: list[AgentConfig] = Field(
        default_factory=list, description="The allow-list of agents available to this process."
    )
    tools: list[ToolConfig] = Field(
        default_factory=list, description="The allow-list of tools available to this process."
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration."
    )
