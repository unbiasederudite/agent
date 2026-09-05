"""Tests for api/app.py's session lifecycle routes — GET and DELETE."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.app import add_exception_handlers, add_session_routes
from agent.core.models.message import Message
from agent.core.services.session_service import SessionService
from agent.core.session_stores.in_memory import InMemorySessionStore


def _client_for(store: InMemorySessionStore) -> TestClient:
    app = FastAPI()
    add_exception_handlers(app)
    add_session_routes(app, SessionService(store))
    return TestClient(app, raise_server_exceptions=False)


async def test_get_session_returns_the_full_stored_history():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    await store.append(
        "researcher",
        session_id,
        [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
    )
    client = _client_for(store)

    response = client.get(f"/v1/agents/researcher/sessions/{session_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hi"
    assert body["messages"][1]["role"] == "assistant"


async def test_get_session_includes_tool_messages_unfiltered():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    await store.append(
        "researcher",
        session_id,
        [Message(role="tool", content="42", tool_call_id="call_1", name="get_current_time")],
    )
    client = _client_for(store)

    response = client.get(f"/v1/agents/researcher/sessions/{session_id}")

    assert response.json()["messages"][0]["role"] == "tool"


def test_get_session_given_unknown_session_returns_404():
    store = InMemorySessionStore()
    client = _client_for(store)

    response = client.get("/v1/agents/researcher/sessions/does-not-exist")

    assert response.status_code == 404


async def test_delete_session_removes_it():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    client = _client_for(store)

    response = client.delete(f"/v1/agents/researcher/sessions/{session_id}")

    assert response.status_code == 204
    assert client.get(f"/v1/agents/researcher/sessions/{session_id}").status_code == 404


def test_delete_session_given_unknown_session_returns_404():
    store = InMemorySessionStore()
    client = _client_for(store)

    response = client.delete("/v1/agents/researcher/sessions/does-not-exist")

    assert response.status_code == 404


async def test_delete_session_given_currently_busy_returns_409():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    client = _client_for(store)

    async with store.busy("researcher", session_id):
        response = client.delete(f"/v1/agents/researcher/sessions/{session_id}")

    assert response.status_code == 409
