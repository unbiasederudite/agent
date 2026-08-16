from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app

_CONFIG_PATH = Path(__file__).parent / "agent_app_config.json"


def test_agent_completions_given_agent_only_uses_its_default_llm_and_persona():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"agent": "pong-bot", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o-mini"
    assert "pong" in body["choices"][0]["message"]["content"].lower()
