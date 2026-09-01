"""Backend-native request/response schemas for the agent-run and registry-listing routes."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class AgentRunRequest(BaseModel):
    """Request body for POST /v1/agents/{agent_name}."""

    model_config = ConfigDict(extra="forbid")

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
        default=None,
        ge=0,
        description=(
            "Overrides the agent's/LLM's configured default, if set. Only the lower bound "
            "is enforced here -- the upper bound is provider-specific (2 for OpenAI, 1 for "
            "Anthropic), so a too-high value is left for the provider itself to reject "
            "rather than guessed at here and potentially validated wrong for the agent's "
            "actual configured model."
        ),
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Overrides the agent's/LLM's configured default, if set.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Overrides the agent's/LLM's configured default, if set.",
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Registered tool names to offer the LLM. Omitted/null uses the agent's "
            "configured tools (or none); an empty list suppresses tools even if the agent "
            "has some; a non-empty list is used as-is, ignoring the agent's own."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Continues an existing conversation with this agent. Omit to start a new one "
            "-- the response's session_id is then a freshly created one to pass on the "
            "next call. A session_id created under a different agent is treated as unknown."
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
    session_id: str = Field(
        description=(
            "The session this response belongs to -- pass it back to continue the conversation."
        )
    )


class SessionHistoryResponse(BaseModel):
    """Response body for GET /v1/agents/{agent_name}/sessions/{session_id}."""

    session_id: str = Field(description="The session this history belongs to.")
    messages: list[Message] = Field(
        description="Every stored message, in order, unfiltered -- every role, including "
        "tool calls and tool results exactly as stored."
    )


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
