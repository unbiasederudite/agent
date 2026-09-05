"""Domain record of one completion execution."""

from pydantic import BaseModel, Field

from agent.core.models.message import Message
from agent.core.models.usage import ZERO_USAGE, Usage


class Run(BaseModel):
    """Domain record of one completion execution."""

    model: str = Field(
        description="The litellm-format provider/model string used for this execution."
    )
    response: Message = Field(description="The LLM's reply message.")
    usage: Usage = Field(description="Token usage for this execution's own turn.")
    supporting_usage: Usage = Field(
        default=ZERO_USAGE,
        description="Token usage from this run's own supporting LLM calls — a guardrail "
        "check, a compaction summary — separate from the turn above.",
    )
    finish_reason: str = Field(
        description='Why generation stopped (e.g. "stop", "length", "content_filter").'
    )
    session_id: str = Field(
        description=(
            "The session this execution belongs to — the given session_id, echoed back, "
            "or a newly created one if none was given."
        )
    )
