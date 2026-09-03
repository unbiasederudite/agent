"""Protocol interface for reasoning-loop implementations."""

from typing import Protocol

from agent.core.models.message import Message
from agent.core.models.turn import Turn
from agent.core.protocols.illm import ILLM
from agent.core.protocols.itool import ITool


class IStrategy(Protocol):
    """Interface for a reasoning loop that turns messages into a final Turn."""

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

        Args:
            messages: The initial message list.
            llm: The LLM to call.
            tools: Tools available to invoke, by name.
            max_iterations: Cap on iterations.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max output tokens.
            max_tool_result_chars: Cap on a single tool result's length.
            max_tool_calls_per_round: Cap on tool calls executed per round.
            max_tool_results_total_chars: Cap on combined tool-result length for the run.

        Returns:
            Turn: the run's aggregate result.
        """
        ...
