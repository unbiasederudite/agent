from pathlib import Path

from fastapi.testclient import TestClient

from agent.api.app import create_app

_MODEL = "openai/gpt-4o-mini"
_CONFIG_PATH = Path(__file__).parent / "app_config.json"


def test_chat_completions_given_real_llm_returns_assistant_reply():
    app = create_app(_CONFIG_PATH)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        },
    )

    assert response.status_code == 200
    assert "pong" in response.json()["choices"][0]["message"]["content"].lower()
