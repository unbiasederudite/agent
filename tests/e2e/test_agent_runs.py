from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app

_CONFIG_PATH = Path(__file__).parent / "agent_app_config.json"


def test_agent_runs_given_agent_uses_its_default_llm_and_persona():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/pong-bot", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o-mini"
    assert "pong" in body["message"]["content"].lower()


def test_list_agents_given_real_config_returns_pong_bot():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.get("/v1/agents")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "pong-bot", "default_llm": "openai/gpt-4o-mini", "tools": []}
    ]
