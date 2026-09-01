"""Protocol interface for tool implementations."""

from typing import Any, Protocol

from pydantic import BaseModel


class ITool(Protocol):
    """Interface for a tool an agent can expose to an LLM.

    `parameters_model` is the schema of record for this tool's arguments: the LLM-facing
    JSON Schema is derived from it (`model_json_schema()`), and an LLM's returned arguments
    are validated against it (`model_validate()`) before `execute()` runs -- see
    `core/strategies/react.py`'s `_execute_call()`. `execute()` is called by whichever
    `IStrategy` implementation the request routes through.
    """

    name: str
    description: str
    parameters_model: type[BaseModel]

    async def execute(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments and return its result as a string."""
        ...
