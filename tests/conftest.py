"""Gate e2e tests behind E2E_TESTS=1."""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip e2e tests unless E2E_TESTS=1 is set."""
    if os.environ.get("E2E_TESTS") == "1":
        return

    skip_e2e = pytest.mark.skip(reason="e2e tests require E2E_TESTS=1")
    for item in items:
        if "e2e" in item.path.parts:
            item.add_marker(skip_e2e)
