"""Domain record of one completion execution."""

from pydantic import BaseModel, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class Run(BaseModel):
    """Domain record of one completion execution."""

    model: str = Field(
        description="The litellm-format provider/model string used for this execution."
    )
    request: list[Message] = Field(description="The messages sent to the LLM.")
    response: Message = Field(description="The LLM's reply message.")
    usage: Usage = Field(description="Token usage for this execution.")
    finish_reason: str = Field(
        description='Why generation stopped (e.g. "stop", "length", "content_filter").'
    )
