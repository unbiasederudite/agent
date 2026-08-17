"""Registry for mapping tool names to ITool implementations."""

from agent.core.exceptions import ToolNotFoundError
from agent.core.protocols.itool import ITool
from agent.core.registries.base import _Registry


class ToolRegistry(_Registry[ITool]):
    """Name-to-instance map of registered ITool implementations.

    `get()` raises `ToolNotFoundError` if the requested name is not registered.
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        super().__init__(ToolNotFoundError)
