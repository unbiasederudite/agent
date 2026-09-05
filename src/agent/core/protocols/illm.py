"""Protocol interfaces for LLM implementations."""

from typing import Any, Protocol

from agent.core.models.completion import Completion
from agent.core.models.message import Message


class ILLM(Protocol):
    """Interface for anything that can turn messages into a completion."""

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        """Send messages to an LLM and return its completion.

        Args:
            messages: The conversation history to send.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max output tokens.
            tools: OpenAI-format function schemas to offer the model.

        Returns:
            Completion: the model's response.
        """
        ...

    def max_input_tokens(self) -> int:
        """Return this model's maximum input token count, via a synchronous local lookup.

        Returns:
            int: the model's maximum input token count.

        Raises:
            LLMError: if the underlying provider has no known limit for this model.
        """
        ...
