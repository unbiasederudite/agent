from datetime import datetime

from agent.core.tools.get_current_time import GetCurrentTimeTool


async def test_execute_returns_parseable_iso_8601_utc_timestamp():
    tool = GetCurrentTimeTool()

    result = await tool.execute()

    datetime.fromisoformat(result)


def test_tool_declares_name_description_and_parameters():
    tool = GetCurrentTimeTool()

    assert tool.name == "get_current_time"
    assert tool.description
    assert tool.parameters == {"type": "object", "properties": {}}
