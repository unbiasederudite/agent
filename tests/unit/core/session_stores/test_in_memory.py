"""Tests for InMemorySessionStore."""

import pytest

from agent.core.exceptions import SessionNotFoundError
from agent.core.models.message import Message
from agent.core.session_stores.in_memory import InMemorySessionStore


async def test_create_returns_an_id_with_empty_history():
    store = InMemorySessionStore()

    session_id = await store.create("researcher")

    assert await store.get("researcher", session_id) == []


async def test_append_then_get_round_trips_messages():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]

    await store.append("researcher", session_id, messages)

    assert await store.get("researcher", session_id) == messages


async def test_append_twice_accumulates_in_order():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    first = [Message(role="user", content="hi")]
    second = [Message(role="assistant", content="hello")]

    await store.append("researcher", session_id, first)
    await store.append("researcher", session_id, second)

    assert await store.get("researcher", session_id) == first + second


async def test_get_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", "does-not-exist")


async def test_append_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.append("researcher", "does-not-exist", [Message(role="user", content="hi")])


async def test_get_given_session_created_under_a_different_agent_raises_session_not_found_error():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")

    with pytest.raises(SessionNotFoundError):
        await store.get("scheduler", session_id)


async def test_create_given_two_calls_returns_different_ids():
    store = InMemorySessionStore()

    first_id = await store.create("researcher")
    second_id = await store.create("researcher")

    assert first_id != second_id


async def test_sessions_under_different_agents_are_independent():
    store = InMemorySessionStore()
    researcher_id = await store.create("researcher")
    scheduler_id = await store.create("scheduler")
    await store.append("researcher", researcher_id, [Message(role="user", content="a")])

    assert await store.get("researcher", researcher_id) == [Message(role="user", content="a")]
    assert await store.get("scheduler", scheduler_id) == []


async def test_replace_overwrites_existing_history():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    await store.append("researcher", session_id, [Message(role="user", content="old")])

    await store.replace("researcher", session_id, [Message(role="assistant", content="summary")])

    assert await store.get("researcher", session_id) == [
        Message(role="assistant", content="summary")
    ]


async def test_replace_given_unknown_session_id_raises_session_not_found_error():
    store = InMemorySessionStore()

    with pytest.raises(SessionNotFoundError):
        await store.replace("researcher", "does-not-exist", [Message(role="user", content="hi")])
