"""OpenAI-compatible message model."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ToolCallFunction(BaseModel):
    """The function half of a tool call, in litellm/OpenAI's nested wire shape."""

    name: str = Field(description="Name of the tool to invoke.")
    arguments: str = Field(
        description=(
            "Raw JSON string of arguments, unparsed -- parsing it against the tool's "
            "declared parameter schema is a future milestone's (the executor's) job, "
            "not this one's."
        )
    )


class ToolCall(BaseModel):
    """One tool invocation requested by the LLM, in litellm/OpenAI's nested wire shape."""

    id: str = Field(description="Unique identifier for this tool call.")
    type: Literal["function"] = Field(default="function", description="The tool call type.")
    function: ToolCallFunction = Field(description="The function to invoke.")


class Message(BaseModel):
    """A single chat message in OpenAI-compatible wire format."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The message's role in the conversation."
    )
    content: str | None = Field(
        default=None,
        description="The message's text content. `None` when `tool_calls` is set instead.",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Tool calls the LLM wants invoked, if any."
    )

    @model_validator(mode="after")
    def _require_content_or_tool_calls(self) -> "Message":
        if self.content is None and not self.tool_calls:
            raise ValueError("either `content` or `tool_calls` must be given")
        return self
