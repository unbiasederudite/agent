"""Backend-native request/response schemas for the agent-run and registry-listing routes."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class AgentRunRequest(BaseModel):
    """Request body for POST /v1/agents/{agent_name}."""

    messages: list[Message] = Field(
        description=(
            "The conversation history to send. `tool_calls` on a message is response-only "
            'and rejected here -- this milestone has no `role: "tool"`/`tool_call_id` '
            "support, so a replayed assistant tool-call turn can't be completed and would "
            "only be rejected by the provider instead."
        )
    )
    model: str | None = Field(
        default=None,
        description=(
            "litellm-format provider/model string overriding the agent's default_llm; must "
            "be declared in the server's config."
        ),
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

    @model_validator(mode="after")
    def _reject_inbound_tool_calls(self) -> "AgentRunRequest":
        if any(m.tool_calls for m in self.messages):
            raise ValueError("`tool_calls` on a request message is not supported")
        return self


class AgentRunResponse(BaseModel):
    """Response body for POST /v1/agents/{agent_name}.

    Deliberately not `core.models.run.Run` reused wholesale: `Run.request` echoes the full
    resolved message list sent to the LLM, including the agent's prepended `system_prompt` --
    exposing that here would leak agent identity content through every response.
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
    default_llm: str = Field(description="The LLM used when a request doesn't override `model`.")
    tools: list[str] = Field(description="Tool names available to this agent by default.")


class ToolSummary(BaseModel):
    """One entry in the GET /v1/tools listing."""

    name: str = Field(description="The tool's lookup key.")
    description: str = Field(description="Human-readable description of what the tool does.")
    parameters: dict[str, Any] = Field(description="JSON schema for the tool's call arguments.")
