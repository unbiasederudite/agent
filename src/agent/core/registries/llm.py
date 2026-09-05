"""Registry for mapping model names to their completion-generating implementations."""

from agent.core.exceptions import LLMNotFoundError
from agent.core.protocols.illm import ILLM
from agent.core.registries.base import _Registry


class LLMRegistry(_Registry[ILLM]):
    """Name-to-instance map of registered model implementations.

    Raises:
        LLMNotFoundError: if the name isn't registered.
    """

    def __init__(self) -> None:
        """Initialize an empty LLM registry."""
        super().__init__(LLMNotFoundError)
