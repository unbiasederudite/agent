"""In-process, non-durable ISessionStore implementation."""

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from agent.core.exceptions import SessionBusyError, SessionNotFoundError
from agent.core.models.message import Message

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    """ISessionStore backed by a process-local dict. Lost on restart.

    `get`/`append`/`create` need no locking of their own: each method body has no `await`
    before it finishes mutating state, so under asyncio's single-threaded cooperative
    scheduling no other coroutine can interleave mid-operation. `lock()` exists for a
    different problem: a read-modify-write that spans more than one call, with an `await`
    in between (e.g. `CompactionService.compact()`'s get-summarize-replace), is not safe
    against a concurrent single-call mutation (e.g. `AgentRunService.run()`'s `append()`)
    on its own -- `lock()` lets callers serialize such a sequence against that pair. Bounded
    by max_sessions (constructor arg) via LRU eviction across _sessions and _locks together
    -- see _evict_if_over_capacity.
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        """Initialize an empty session store.

        Args:
            max_sessions: Caps how many distinct sessions are kept at once. `None` means
                unbounded. Once over the cap, the least-recently-touched session is
                evicted to make room for a new one -- see `_evict_if_over_capacity`.
        """
        self._sessions: OrderedDict[tuple[str, str], list[Message]] = OrderedDict()
        self._locks: OrderedDict[tuple[str, str], asyncio.Lock] = OrderedDict()
        self._busy: set[tuple[str, str]] = set()
        self._max_sessions = max_sessions

    def _evict_if_over_capacity(self, protect: tuple[str, str]) -> None:
        """Evict the oldest-touched session once `_sessions` exceeds `_max_sessions`.

        Skips any key whose lock is currently held or whose `busy()` marker is set --
        evicting an actively in-flight session would let a second caller silently create
        a fresh `Lock` for the same key while the original is still logically held,
        breaking mutual exclusion. Also skips `protect`, the session `create()` just
        inserted: without this, a store already at capacity with every existing entry
        held/busy would evict the brand-new session instead of failing open, since it is
        the only remaining candidate with no lock/busy marker of its own yet. Fails open
        if every other existing entry is currently held/busy (an extreme edge case): the
        cap is a steady-state guarantee, not a hard admission-control limit, so a new
        session is never rejected because the store is transiently over capacity due to
        legitimate in-flight work.
        """
        if self._max_sessions is None or len(self._sessions) <= self._max_sessions:
            return
        for key in list(self._sessions.keys()):
            if key == protect:
                continue
            lock = self._locks.get(key)
            if key in self._busy or (lock is not None and lock.locked()):
                continue
            del self._sessions[key]
            self._locks.pop(key, None)
            return

    async def create(self, agent: str) -> str:
        """Allocate a new session under `agent`, with empty history, and return its id."""
        session_id = uuid.uuid4().hex
        key = (agent, session_id)
        self._sessions[key] = []
        self._evict_if_over_capacity(protect=key)
        return session_id

    async def get(self, agent: str, session_id: str) -> list[Message]:
        """Return the stored history for `(agent, session_id)`.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        self._sessions.move_to_end(key)
        return list(self._sessions[key])

    async def append(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Extend the stored history for `(agent, session_id)` with `messages`, in order.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        self._sessions[key].extend(messages)
        self._sessions.move_to_end(key)

    async def replace(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Overwrite the stored history for `(agent, session_id)` with `messages` entirely.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        self._sessions[key] = list(messages)

    @asynccontextmanager
    async def lock(self, agent: str, session_id: str) -> AsyncIterator[None]:
        """`ISessionStore.lock()` -- see that docstring.

        Lock creation itself needs no synchronization: no `await` happens between the dict
        lookup and insertion below, so no other coroutine can interleave, the same reasoning
        this class already relies on for its other methods never needing a lock of their own.
        """
        key = (agent, session_id)
        session_lock = self._locks.get(key)
        if session_lock is None:
            session_lock = asyncio.Lock()
            self._locks[key] = session_lock
        else:
            self._locks.move_to_end(key)
        if session_lock.locked():
            logger.debug("waiting for session lock (%s, %s), currently held", agent, session_id)
        try:
            async with session_lock:
                yield
        finally:
            if key not in self._sessions:
                self._locks.pop(key, None)

    @asynccontextmanager
    async def busy(self, agent: str, session_id: str) -> AsyncIterator[None]:
        """`ISessionStore.busy()` -- see that docstring.

        A plain `set`, not an `asyncio.Lock`: this is an immediate test-and-set, never a
        wait. No `await` between the membership check and the insert below keeps it
        race-free under asyncio's cooperative scheduling, same reasoning `lock()`'s own
        lazy creation already relies on.
        """
        key = (agent, session_id)
        if key in self._busy:
            raise SessionBusyError(f"session '{session_id}' for agent '{agent}' is busy")
        self._busy.add(key)
        try:
            yield
        finally:
            self._busy.discard(key)

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and all its stored history.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
        del self._sessions[key]
        self._locks.pop(key, None)
