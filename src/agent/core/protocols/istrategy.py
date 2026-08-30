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
        max_tool_result_chars: int | None = None,
        max_tool_calls_per_round: int | None = None,
        max_tool_results_total_chars: int | None = None,
    ) -> Turn:
        """Run the loop and return the final Turn.

        `tools` maps offered tool names to their instances -- the exhaustive set this
        strategy may invoke, already resolved by the caller. `max_iterations` bounds
        however this strategy defines "a round." `temperature`/`top_p`/`max_tokens` are
        forwarded to every LLM call this strategy makes, same meaning as `ILLM.complete`'s
        own params. `max_tool_result_chars`, if given, caps how much of a tool's result
        content is fed back into the message list -- a longer result is truncated with a
        trailing marker; `None` means uncapped. `max_tool_calls_per_round`, if given, caps
        how many tool calls from one LLM response actually execute -- the excess are
        skipped, each replaced with a short error result saying so, so the LLM can adjust
        next round; `None` means uncapped. `max_tool_results_total_chars`, if given, caps
        the combined length of every tool result's content across the whole run, all rounds
        together -- once that budget is spent, every further result is replaced with a short
        omission marker instead of its real content; `None` means uncapped.
        """
        ...
