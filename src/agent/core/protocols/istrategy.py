"""Protocol interface for reasoning-loop implementations."""

from typing import Protocol

from agent.core.models.message import Message
from agent.core.models.turn import Turn
from agent.core.protocols.illm import ILLM
from agent.core.protocols.itool import ITool


class IStrategy(Protocol):
    """Interface for a reasoning loop that turns messages into a final Turn.

    Owns everything about how (and whether) tools get offered to and invoked by the LLM;
    `messages` is the already-resolved initial list (system prompt + user turn). `tools` is
    already resolved to instances -- the caller owns name resolution against the
    `ToolRegistry`, so a strategy never sees a tool it wasn't explicitly given.
    """

    async def run(
        self,
        messages: list[Message],
        llm: ILLM,
        tools: dict[str, ITool],
        max_iterations: int,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Turn:
        """Run the loop and return the final Turn.

        `tools` maps offered tool names to their instances -- the exhaustive set this
        strategy may invoke, already resolved by the caller. `max_iterations` bounds
        however this strategy defines "a round." `temperature`/`top_p`/`max_tokens` are
        forwarded to every LLM call this strategy makes, same meaning as `ILLM.complete`'s
        own params.
        """
        ...
