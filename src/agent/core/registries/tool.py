"""Registry for mapping tool names to their implementations."""

from agent.core.exceptions import ToolNotFoundError
from agent.core.protocols.itool import ITool
from agent.core.registries.base import _Registry


class ToolRegistry(_Registry[ITool]):
    """Name-to-instance map of registered tool implementations.

    Raises:
        ToolNotFoundError: if the name isn't registered.
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        super().__init__(ToolNotFoundError)

    def get_many(self, names: list[str]) -> dict[str, ITool]:
        """Return the registered tools for `names`, keyed by each tool's own `name`.

        Args:
            names: Lookup keys.

        Returns:
            dict[str, ITool]: the registered tools, keyed by `ITool.name`.

        Raises:
            ToolNotFoundError: if any name isn't registered.
        """
        return {tool.name: tool for tool in (self.get(name) for name in names)}
