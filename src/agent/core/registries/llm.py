"""Registry for mapping model names to ILLM implementations."""

from agent.core.exceptions import LLMNotFoundError
from agent.core.protocols.illm import ILLM


class LLMRegistry:
    """Name-to-instance map of registered ILLM implementations."""

    def __init__(self) -> None:
        """Initialize an empty LLM registry."""
        self._llms: dict[str, ILLM] = {}

    def register(self, name: str, llm: ILLM) -> None:
        """Register an LLM instance under `name`."""
        self._llms[name] = llm

    def get(self, name: str) -> ILLM:
        """Return the registered LLM for `name`.

        Raises:
            LLMNotFoundError: if `name` is not registered.
        """
        try:
            return self._llms[name]
        except KeyError:
            raise LLMNotFoundError(name) from None
