"""OpenAI-compatible message model."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ToolCallFunction(BaseModel):
    """The function half of a tool call, in litellm/OpenAI's nested wire shape."""

    name: str = Field(description="Name of the tool to invoke.")
    arguments: str = Field(
        description=(
            "Raw JSON string of arguments. The executing IStrategy parses this with "
            "json.loads, but nothing validates the result against the tool's declared "
            "`ITool.parameters` schema."
        )
    )


class ToolCall(BaseModel):
    """One tool invocation requested by the LLM, in litellm/OpenAI's nested wire shape."""

    id: str = Field(description="Unique identifier for this tool call.")
    type: Literal["function"] = Field(default="function", description="The tool call type.")
    function: ToolCallFunction = Field(description="The function to invoke.")


class Message(BaseModel):
    """A single chat message in OpenAI-compatible wire format."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The message's role in the conversation."
    )
    content: str | None = Field(
        default=None,
        description="The message's text content. `None` when `tool_calls` is set instead.",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Tool calls the LLM wants invoked, if any."
    )
    tool_call_id: str | None = Field(
        default=None,
        description='The `ToolCall.id` this message answers. Required when `role="tool"`.',
    )
    name: str | None = Field(
        default=None,
        description='The tool name this message answers. Required when `role="tool"`.',
    )

    @model_validator(mode="after")
    def _validate_role_shape(self) -> "Message":
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError('`tool_call_id` is required when role="tool"')
            if self.name is None:
                raise ValueError('`name` is required when role="tool"')
            if self.content is None:
                raise ValueError('`content` is required when role="tool"')
            if self.tool_calls:
                raise ValueError('`tool_calls` must not be set when role="tool"')
        elif self.content is None and not self.tool_calls:
            raise ValueError("either `content` or `tool_calls` must be given")
        return self
