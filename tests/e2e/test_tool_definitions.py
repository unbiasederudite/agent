from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app

_CONFIG_PATH = Path(__file__).parent / "tool_app_config.json"


def test_tool_definitions_given_agent_with_tools_returns_tool_call():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/clock-bot",
        json={
            "messages": [{"role": "user", "content": "Use your tool to check the current time."}]
        },
    )

    assert response.status_code == 200
    body = response.json()
    tool_calls = body["message"]["tool_calls"]
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "get_current_time"
    assert body["finish_reason"] == "tool_calls"
