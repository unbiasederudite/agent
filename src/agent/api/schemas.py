"""Backend-native request/response schemas for the agent-run and registry-listing routes."""

from typing import Any

from pydantic import BaseModel, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class AgentRunRequest(BaseModel):
    """Request body for POST /v1/agents/{agent_name}."""

    message: str = Field(description="The user's message to send to the agent.")
    model: str | None = Field(
        default=None,
        description=(
            "litellm-format provider/model string overriding the agent's configured `model`; "
            "be declared in the server's config."
        ),
    )
    strategy: str | None = Field(
        default=None,
        description="Reasoning strategy name overriding the agent's configured strategy.",
    )
    temperature: float | None = Field(
        default=None, description="Overrides the agent's/LLM's configured default, if set."
    )
    top_p: float | None = Field(
        default=None, description="Overrides the agent's/LLM's configured default, if set."
    )
    max_tokens: int | None = Field(
        default=None, description="Overrides the agent's/LLM's configured default, if set."
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Registered tool names to offer the LLM. Omitted/null uses the agent's "
            "configured tools (or none); an empty list suppresses tools even if the agent "
            "has some; a non-empty list is used as-is, ignoring the agent's own."
        ),
    )


class AgentRunResponse(BaseModel):
    """Response body for POST /v1/agents/{agent_name}.

    Its own model rather than reusing `core.models.run.Run` directly, keeping the
    API-facing DTO independent of the internal domain model.
    """

    model: str = Field(
        description="The litellm-format provider/model string that ran this request."
    )
    message: Message = Field(description="The generated reply message.")
    usage: Usage = Field(description="Token usage for this run.")
    finish_reason: str = Field(description="Why generation stopped.")


class AgentSummary(BaseModel):
    """One entry in the GET /v1/agents listing."""

    name: str = Field(description="The agent's lookup key.")
    model: str = Field(description="The LLM used when a request doesn't override `model`.")
    strategy: str = Field(
        description="The reasoning strategy used when a request doesn't override `strategy`."
    )
    tools: list[str] = Field(description="Tool names available to this agent by default.")


class ToolSummary(BaseModel):
    """One entry in the GET /v1/tools listing."""

    name: str = Field(description="The tool's lookup key.")
    description: str = Field(description="Human-readable description of what the tool does.")
    parameters: dict[str, Any] = Field(description="JSON schema for the tool's call arguments.")
