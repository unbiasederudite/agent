import pytest

from agent.core.exceptions import StrategyNotFoundError
from agent.core.registries.strategy import StrategyRegistry


class _FakeStrategy:
    async def run(self, *args: object, **kwargs: object) -> None:
        return None


def test_strategy_registry_given_registered_name_returns_instance():
    registry = StrategyRegistry()
    strategy = _FakeStrategy()

    registry.register("react", strategy)

    assert registry.get("react") is strategy


def test_strategy_registry_given_unregistered_name_raises_strategy_not_found_error():
    registry = StrategyRegistry()

    with pytest.raises(StrategyNotFoundError):
        registry.get("missing")


def test_strategy_registry_all_returns_every_registered_name_to_instance():
    registry = StrategyRegistry()
    strategy = _FakeStrategy()
    registry.register("react", strategy)

    assert registry.all() == {"react": strategy}
