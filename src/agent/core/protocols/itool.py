"""Protocol interface for tool implementations."""

from typing import Any, Protocol


class ITool(Protocol):
    """Interface for a tool an agent can expose to an LLM.

    `execute()` is called by whichever `IStrategy` implementation the request routes
    through.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments and return its result as a string."""
        ...
