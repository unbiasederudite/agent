"""OpenAI-compatible message model."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message in OpenAI-compatible wire format."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The message's role in the conversation."
    )
    content: str = Field(description="The message's text content.")
