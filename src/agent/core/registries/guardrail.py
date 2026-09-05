"""Registry for mapping guardrail names to their implementations."""

from agent.core.exceptions import GuardrailNotFoundError
from agent.core.protocols.iguardrail import IGuardrail
from agent.core.registries.base import _Registry


class GuardrailRegistry(_Registry[IGuardrail]):
    """Name-to-instance map of registered guardrail implementations.

    Raises:
        GuardrailNotFoundError: if the name isn't registered.
    """

    def __init__(self) -> None:
        """Initialize an empty guardrail registry."""
        super().__init__(GuardrailNotFoundError)

    def get_many(self, names: list[str]) -> list[IGuardrail]:
        """Return the registered guardrails for `names`, in order.

        Args:
            names: Lookup keys, in the order they should run.

        Returns:
            list[IGuardrail]: the registered guardrails, in `names` order.

        Raises:
            GuardrailNotFoundError: if any name isn't registered.
        """
        return [self.get(name) for name in names]
