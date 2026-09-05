"""Generic name-to-instance registry base for typed registries."""

from collections.abc import Callable


class _Registry[T]:
    """Name-to-instance map that raises a caller-supplied exception on a missing name."""

    def __init__(self, not_found_error: Callable[[str], Exception]) -> None:
        """Initialize an empty registry.

        Args:
            not_found_error: Exception type raised on a missing name.
        """
        self._items: dict[str, T] = {}
        self._not_found_error = not_found_error

    def register(self, name: str, item: T) -> None:
        """Register `item` under `name`.

        Args:
            name: Lookup key.
            item: Instance to register.
        """
        self._items[name] = item

    def get(self, name: str) -> T:
        """Return the registered item for `name`.

        Args:
            name: Lookup key.

        Returns:
            T: the registered item.

        Raises:
            The registered `not_found_error` exception.
        """
        try:
            return self._items[name]
        except KeyError:
            raise self._not_found_error(name) from None

    def all(self) -> dict[str, T]:
        """Return a copy of every registered `name -> instance` mapping.

        Returns:
            dict[str, T]: the registered instances, by name.
        """
        return dict(self._items)
