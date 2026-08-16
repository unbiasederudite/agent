"""Pydantic config models for application startup."""

from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
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


class AppConfig(BaseModel):
    """Root startup configuration, loaded once from a JSON file."""

    llms: list[LLMConfig] = Field(description="The allow-list of LLMs available to this process.")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration."
    )
