"""Generic name-to-instance registry base, shared by LLMRegistry and AgentRegistry."""

from collections.abc import Callable


class _Registry[T]:
    """Name-to-instance map that raises a caller-supplied exception on a missing name."""

    def __init__(self, not_found_error: Callable[[str], Exception]) -> None:
        """Initialize an empty registry.

        Args:
            not_found_error: Exception type to raise (called with the missing name) when
                `get()` is called for an unregistered name.
        """
        self._items: dict[str, T] = {}
        self._not_found_error = not_found_error

    def register(self, name: str, item: T) -> None:
        """Register `item` under `name`."""
        self._items[name] = item

    def get(self, name: str) -> T:
        """Return the registered item for `name`.

        Raises:
            The exception passed as `not_found_error` at construction, if `name` is not
            registered.
        """
        try:
            return self._items[name]
        except KeyError:
            raise self._not_found_error(name) from None
