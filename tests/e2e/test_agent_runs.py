from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app

_CONFIG_PATH = Path(__file__).parent / "agent_app_config.json"


def test_agent_runs_given_agent_uses_its_model_and_persona():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post("/v1/agents/pong-bot", json={"message": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "openai/gpt-4o-mini"
    assert "pong" in body["message"]["content"].lower()


def test_list_agents_given_real_config_returns_both_agents():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.get("/v1/agents")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "pong-bot", "model": "openai/gpt-4o-mini", "strategy": "react", "tools": []},
        {"name": "memory-bot", "model": "openai/gpt-4o-mini", "strategy": "react", "tools": []},
    ]


def test_agent_runs_given_session_id_remembers_the_earlier_turn():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    first = client.post(
        "/v1/agents/memory-bot",
        json={"message": "My favorite color is teal. Just say OK."},
    )
    session_id = first.json()["session_id"]

    second = client.post(
        "/v1/agents/memory-bot",
        json={"message": "What's my favorite color?", "session_id": session_id},
    )

    assert second.status_code == 200
    assert "teal" in second.json()["message"]["content"].lower()


def test_agent_runs_given_unknown_session_id_returns_404():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post(
        "/v1/agents/memory-bot", json={"message": "hi", "session_id": "does-not-exist"}
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"
