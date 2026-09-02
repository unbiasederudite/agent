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
    assert body["usage"]["cost_usd"] is not None


def test_list_agents_given_real_config_returns_all_agents():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.get("/v1/agents")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "pong-bot", "model": "openai/gpt-4o-mini", "strategy": "react", "tools": []},
        {"name": "memory-bot", "model": "openai/gpt-4o-mini", "strategy": "react", "tools": []},
        {"name": "compacting-bot", "model": "openai/gpt-4o-mini", "strategy": "react", "tools": []},
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


def test_agent_runs_given_tiny_token_budget_compacts_and_still_recalls_early_fact():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    first = client.post(
        "/v1/agents/compacting-bot",
        json={"message": "My favorite color is teal. Just say OK."},
    )
    session_id = first.json()["session_id"]

    # Each of these real turns pushes the session further over the tiny configured budget,
    # forcing at least one real compaction pass before the final question below.
    last_filler_response = None
    for i in range(8):
        last_filler_response = client.post(
            "/v1/agents/compacting-bot",
            json={
                "message": f"Tell me a short, made-up fact about the number {i}.",
                "session_id": session_id,
            },
        )

    final = client.post(
        "/v1/agents/compacting-bot",
        json={"message": "What's my favorite color?", "session_id": session_id},
    )

    assert final.status_code == 200
    assert "teal" in final.json()["message"]["content"].lower()
    # A plain, ever-growing history would send flat-to-more prompt_tokens each turn. A drop
    # here is near-impossible without a real compaction pass having shrunk stored history in
    # between -- this is what actually discriminates "compaction ran" from "it didn't need to".
    assert last_filler_response is not None
    assert (
        final.json()["usage"]["prompt_tokens"]
        < last_filler_response.json()["usage"]["prompt_tokens"]
    )
