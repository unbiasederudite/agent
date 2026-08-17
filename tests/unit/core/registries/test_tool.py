import pytest

from agent.core.exceptions import ToolNotFoundError
from agent.core.registries.tool import ToolRegistry
from agent.core.tools.get_current_time import GetCurrentTimeTool


def test_tool_registry_get_given_registered_name_returns_tool():
    registry = ToolRegistry()
    tool = GetCurrentTimeTool()
    registry.register("get_current_time", tool)

    assert registry.get("get_current_time") is tool


def test_tool_registry_get_given_unregistered_name_raises_tool_not_found_error():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("missing")


def test_tool_registry_all_given_registered_tools_returns_name_to_instance_mapping():
    registry = ToolRegistry()
    tool = GetCurrentTimeTool()
    registry.register("get_current_time", tool)

    assert registry.all() == {"get_current_time": tool}


def test_tool_registry_all_given_none_registered_returns_empty_dict():
    registry = ToolRegistry()

    assert registry.all() == {}
