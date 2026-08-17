"""Protocol interface for tool implementations."""

from typing import Any, Protocol


class ITool(Protocol):
    """Interface for a tool an agent can expose to an LLM.

    `execute()` is declared now so a concrete `ITool` is a real implementation, not
    just a schema -- but nothing calls it this milestone. `tool_calls` an LLM returns
    are passed through to the client unexecuted; execution is a future milestone's
    concern, once `core/strategies/` exists to own the loop.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments and return its result as a string."""
        ...
