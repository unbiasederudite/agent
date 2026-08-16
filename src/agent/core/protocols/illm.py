"""Protocol interfaces for LLM implementations."""

from typing import Protocol

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
    ) -> Completion:
        """Send messages to an LLM and return its completion.

        `temperature`/`top_p`/`max_tokens` of `None` mean "use this implementation's
        configured default," not "omit this param from consideration entirely."
        """
        ...
