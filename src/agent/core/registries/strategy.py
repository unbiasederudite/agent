"""Registry for mapping strategy names to their reasoning-loop implementations."""

from agent.core.exceptions import StrategyNotFoundError
from agent.core.protocols.istrategy import IStrategy
from agent.core.registries.base import _Registry


class StrategyRegistry(_Registry[IStrategy]):
    """Name-to-instance map of registered reasoning-strategy implementations.

    Raises:
        StrategyNotFoundError: if the name isn't registered.
    """

    def __init__(self) -> None:
        """Initialize an empty strategy registry."""
        super().__init__(StrategyNotFoundError)
