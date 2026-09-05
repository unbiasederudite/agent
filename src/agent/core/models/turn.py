"""Aggregate result of one reasoning-loop run."""

from pydantic import BaseModel, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class Turn(BaseModel):
    """Everything one reasoning-loop run generated."""

    messages: list[Message] = Field(
        min_length=1,
        description=(
            "Every message generated during this run, in order: assistant tool-call turns, "
            'role="tool" results, and the final assistant answer as the last element. Never '
            "includes the input `messages` the strategy was called with."
        ),
    )
    usage: Usage = Field(description="Token usage summed across every LLM call this run made.")
    final_total_tokens: int = Field(
        description="total_tokens of the last individual LLM call this run made, not "
        "summed across the run."
    )
    finish_reason: str = Field(
        description="The finish_reason of whichever LLM call produced the final message."
    )

    @property
    def message(self) -> Message:
        """The final answer — always `messages[-1]`. A run always generates at least one.

        Returns:
            Message: the final answer.
        """
        return self.messages[-1]
