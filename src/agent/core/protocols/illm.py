"""Protocol interfaces for LLM implementations."""

from typing import Protocol

from agent.core.models.completion import Completion
from agent.core.models.message import Message


class ILLM(Protocol):
    """Interface for anything that can turn messages into a completion."""

    async def complete(self, messages: list[Message]) -> Completion:
        """Send messages to an LLM and return its completion."""
        ...
