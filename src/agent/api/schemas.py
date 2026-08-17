"""OpenAI-compatible request/response schemas for chat completions."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatToolCallFunction(BaseModel):
    """The function half of an OpenAI-compatible tool call."""

    name: str = Field(description="Name of the tool to invoke.")
    arguments: str = Field(description="Raw JSON string of arguments, unparsed.")


class ChatToolCall(BaseModel):
    """An OpenAI-compatible tool call, as seen over the wire."""

    id: str = Field(description="Unique identifier for this tool call.")
    type: Literal["function"] = Field(default="function", description="The tool call type.")
    function: ChatToolCallFunction = Field(description="The function to invoke.")


class ChatMessage(BaseModel):
    """An OpenAI-compatible chat message, as seen over the wire."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The message's role in the conversation."
    )
    content: str | None = Field(
        default=None, description="The message's text content. `None` when `tool_calls` is set."
    )
    tool_calls: list[ChatToolCall] | None = Field(
        default=None, description="Tool calls the assistant wants invoked, if any."
    )

    @model_validator(mode="after")
    def _require_content_or_tool_calls(self) -> "ChatMessage":
        if self.content is None and not self.tool_calls:
            raise ValueError("either `content` or `tool_calls` must be given")
        return self


class ChatCompletionRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""

    agent: str | None = Field(
        default=None, description="Name of a registered agent to route this request through."
    )
    model: str | None = Field(
        default=None,
        description=(
            "litellm-format provider/model string to route this request to; must be "
            "declared in the server's config. Defaults to the selected agent's "
            "default_llm if omitted."
        ),
    )
    messages: list[ChatMessage] = Field(
        description=(
            "The conversation history to send. `tool_calls` on a message is response-only "
            'and rejected here -- this milestone has no `role: "tool"`/`tool_call_id` '
            "support, so a replayed assistant tool-call turn can't be completed and would "
            "only be rejected by the provider instead."
        )
    )
    temperature: float | None = Field(
        default=None, description="Overrides the agent's/LLM's configured default, if set."
    )
    top_p: float | None = Field(
        default=None, description="Overrides the agent's/LLM's configured default, if set."
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description=(
            "Overrides the agent's/LLM's configured default, if set. Named to match OpenAI's "
            "current field -- the older `max_tokens` is deprecated upstream and not accepted "
            "here."
        ),
    )
    tools: list[str] | None = Field(
        default=None,
        description=(
            "Registered tool names to offer the LLM. Omitted/null uses the selected "
            "agent's configured tools (or none); an empty list suppresses tools even if "
            "the agent has some; a non-empty list is used as-is, ignoring the agent's own."
        ),
    )

    @model_validator(mode="after")
    def _require_agent_or_model(self) -> "ChatCompletionRequest":
        if self.agent is None and self.model is None:
            raise ValueError("either `agent` or `model` must be given")
        return self

    @model_validator(mode="after")
    def _reject_inbound_tool_calls(self) -> "ChatCompletionRequest":
        if any(m.tool_calls for m in self.messages):
            raise ValueError("`tool_calls` on a request message is not supported")
        return self


class ChatCompletionChoice(BaseModel):
    """One completion choice in a chat.completion response."""

    index: int = Field(description="Position of this choice in the choices list.")
    message: ChatMessage = Field(description="The generated reply message.")
    finish_reason: str = Field(description="Why generation stopped.")


class ChatCompletionUsage(BaseModel):
    """Token usage in a chat.completion response."""

    prompt_tokens: int = Field(description="Number of tokens in the input.")
    completion_tokens: int = Field(description="Number of tokens in the generated output.")
    total_tokens: int = Field(description="Total tokens consumed (prompt + completion).")


class ChatCompletionResponse(BaseModel):
    """Response body for POST /v1/chat/completions."""

    id: str = Field(description="Unique identifier for this completion.")
    object: Literal["chat.completion"] = Field(
        default="chat.completion", description="The object type."
    )
    created: int = Field(description="Unix timestamp of when the completion was created.")
    model: str = Field(description="The model that generated this completion.")
    choices: list[ChatCompletionChoice] = Field(description="The generated completion choices.")
    usage: ChatCompletionUsage = Field(description="Token usage for this request.")


class ErrorDetail(BaseModel):
    """OpenAI-compatible error detail."""

    message: str = Field(description="A human-readable error message.")
    type: str = Field(description='The error category, e.g. "invalid_request_error".')
    param: str | None = Field(
        default=None, description="The request field that caused the error, if any."
    )
    code: str | None = Field(
        default=None, description="A short machine-readable error code, if any."
    )


class ErrorResponse(BaseModel):
    """OpenAI-compatible error envelope for POST /v1/chat/completions."""

    error: ErrorDetail = Field(description="The error details.")
