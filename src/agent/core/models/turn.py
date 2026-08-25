"""Aggregate result of one IStrategy.run() call."""

from pydantic import BaseModel, Field

from agent.core.models.message import Message
from agent.core.models.usage import Usage


class Turn(BaseModel):
    """Everything one IStrategy.run() call generated -- not the input it was given.

    Distinct from `Completion` (`core/models/completion.py`), which is the result of a
    single `ILLM.complete()` wire call. A strategy's run spans however many of those calls
    it makes; `Turn` is the aggregate across all of them.
    """

    messages: list[Message] = Field(
        min_length=1,
        description=(
            "Every message generated during this run, in order: assistant tool-call turns, "
            'role="tool" results, and the final assistant answer as the last element. Never '
            "includes the input `messages` the strategy was called with."
        ),
    )
    usage: Usage = Field(description="Token usage summed across every LLM call this run made.")
    finish_reason: str = Field(
        description=(
            "The finish_reason of whichever LLM call produced the final message -- never "
            "invented, never aggregated (unlike usage)."
        )
    )

    @property
    def message(self) -> Message:
        """The final answer -- always `messages[-1]`. A run always generates at least one."""
        return self.messages[-1]
