import logging

import pytest

from agent.core.exceptions import SessionBusyError, SessionNotFoundError
from agent.core.models.message import Message
from agent.core.services.session_service import SessionService
from agent.core.session_stores.in_memory import InMemorySessionStore


class _FakeCompactionService:
    def __init__(self) -> None:
        self.forget_calls: list[tuple[str, str]] = []

    def forget(self, agent: str, session_id: str) -> None:
        self.forget_calls.append((agent, session_id))


async def test_get_history_returns_stored_messages():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    await store.append("researcher", session_id, [Message(role="user", content="hi")])
    service = SessionService(store, None)

    messages = await service.get_history("researcher", session_id)

    assert messages == [Message(role="user", content="hi")]


async def test_get_history_given_unknown_session_raises_session_not_found_error():
    service = SessionService(InMemorySessionStore(), None)

    with pytest.raises(SessionNotFoundError):
        await service.get_history("researcher", "does-not-exist")


async def test_delete_removes_the_session():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store, None)

    await service.delete("researcher", session_id)

    with pytest.raises(SessionNotFoundError):
        await store.get("researcher", session_id)


async def test_delete_given_unknown_session_raises_session_not_found_error():
    service = SessionService(InMemorySessionStore(), None)

    with pytest.raises(SessionNotFoundError):
        await service.delete("researcher", "does-not-exist")


async def test_delete_given_currently_busy_raises_session_busy_error():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store, None)

    async with store.busy("researcher", session_id):
        with pytest.raises(SessionBusyError):
            await service.delete("researcher", session_id)


async def test_delete_forgets_the_compaction_estimate():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    compaction_service = _FakeCompactionService()
    service = SessionService(store, compaction_service)

    await service.delete("researcher", session_id)

    assert compaction_service.forget_calls == [("researcher", session_id)]


async def test_delete_given_no_compaction_service_does_not_raise():
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store, None)

    await service.delete("researcher", session_id)  # must not raise


async def test_delete_given_unknown_session_does_not_forget_compaction_estimate():
    compaction_service = _FakeCompactionService()
    service = SessionService(InMemorySessionStore(), compaction_service)

    with pytest.raises(SessionNotFoundError):
        await service.delete("researcher", "does-not-exist")

    assert compaction_service.forget_calls == []


async def test_delete_logs_info(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent.core.services.session_service")
    store = InMemorySessionStore()
    session_id = await store.create("researcher")
    service = SessionService(store, None)

    await service.delete("researcher", session_id)

    assert any("session deleted" in r.message for r in caplog.records)


async def test_delete_given_unknown_session_does_not_log_success(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="agent.core.services.session_service")
    service = SessionService(InMemorySessionStore(), None)

    with pytest.raises(SessionNotFoundError):
        await service.delete("researcher", "does-not-exist")

    assert caplog.records == []
