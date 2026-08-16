"""Registry for mapping agent names to AgentConfig."""

from agent.core.exceptions import AgentNotFoundError
from agent.core.models.config import AgentConfig
from agent.core.registries.base import _Registry


class AgentRegistry(_Registry[AgentConfig]):
    """Name-to-instance map of registered AgentConfig entries.

    `get()` raises `AgentNotFoundError` if the requested name is not registered.
    """

    def __init__(self) -> None:
        """Initialize an empty agent registry."""
        super().__init__(AgentNotFoundError)
