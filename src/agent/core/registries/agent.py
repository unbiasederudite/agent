"""Registry for mapping agent names to their configuration."""

from agent.core.exceptions import AgentNotFoundError
from agent.core.models.config import AgentConfig
from agent.core.registries.base import _Registry


class AgentRegistry(_Registry[AgentConfig]):
    """Name-to-instance map of registered agent configuration entries.

    Raises:
        AgentNotFoundError: if the name isn't registered.
    """

    def __init__(self) -> None:
        """Initialize an empty agent registry."""
        super().__init__(AgentNotFoundError)
