"""Registry for mapping strategy names to IStrategy implementations."""

from agent.core.exceptions import StrategyNotFoundError
from agent.core.protocols.istrategy import IStrategy
from agent.core.registries.base import _Registry


class StrategyRegistry(_Registry[IStrategy]):
    """Name-to-instance map of registered IStrategy implementations.

    `get()` raises `StrategyNotFoundError` if the requested name is not registered.
    """

    def __init__(self) -> None:
        """Initialize an empty strategy registry."""
        super().__init__(StrategyNotFoundError)
