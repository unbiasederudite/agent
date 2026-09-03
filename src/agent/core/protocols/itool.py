"""Protocol interface for tool implementations."""

from typing import Any, Protocol

from pydantic import BaseModel


class ITool(Protocol):
    """Interface for a tool an agent can expose to an LLM."""

    name: str  # The tool's lookup key, matching a code-level implementation.
    description: str  # Human-readable description of what the tool does.
    parameters_model: type[BaseModel]  # Pydantic model validating this tool's call arguments.

    async def execute(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments and return its result as a string.

        Returns:
            str: the tool's result, for the LLM to read.
        """
        ...
