from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app

_CONFIG_PATH = Path(__file__).parent / "tool_app_config.json"


def test_tool_definitions_given_agent_with_tools_executes_the_tool_and_answers():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/clock-bot", json={"message": "Use your tool to check the current time."}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["tool_calls"] is None
    assert body["message"]["content"] is not None
    assert body["finish_reason"] == "stop"
