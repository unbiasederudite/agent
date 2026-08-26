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


class StrategyConfig(BaseModel):
    """One entry in the startup strategy allow-list.

    `name` must match a code-level strategy implementation (see `core/factories/app.py`)
    and is also the StrategyRegistry lookup key.
    """

    name: str = Field(
        description="The strategy's lookup key, matching a code-level implementation."
    )


class AgentConfig(SamplingDefaults):
    """One entry in the startup agent allow-list.

    `name` is the AgentRegistry lookup key. `system_prompt` is unconditionally prepended
    as the leading system message whenever this agent is selected -- it is identity
    content, not a request-overridable value. `model` must match a declared
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
    model: str = Field(
        description="The LLMConfig.model used when the request doesn't override `model`."
    )
    strategy: str = Field(
        description=(
            "The StrategyRegistry lookup key used when the request doesn't override `strategy`."
        )
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names available to this agent by default, unless the request "
            "overrides them (see AgentRunService.run's tri-state `tools` resolution)."
        ),
    )
    max_tool_iterations: int = Field(
        default=10,
        ge=1,
        description="Caps rounds of call-LLM/maybe-call-a-tool a tool-calling strategy may run.",
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
    strategies: list[StrategyConfig] = Field(
        default_factory=list,
        description="The allow-list of reasoning strategies available to this process.",
    )
    base_prompt: str | None = Field(
        default=None,
        description=(
            "Prepended before every agent's own `system_prompt`, merged into the same "
            "leading system message -- not a second one. Shared instructions across every "
            "agent in this deployment (e.g. house style, compliance rules); omit if there's "
            "nothing to share."
        ),
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration."
    )
