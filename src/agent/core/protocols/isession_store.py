"""Protocol interface for per-conversation message history storage."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from agent.core.models.message import Message


class ISessionStore(Protocol):
    """Interface for per-conversation message history storage.

    Async even though the only current implementation (`InMemorySessionStore`) does no
    I/O -- a durable backend (Redis/SQLite) is the reason this is a protocol at all, and
    that backend will need to await. Keyed by `(agent, session_id)` together, not
    `session_id` alone: a session is locked to the agent it was created under, so a lookup
    under the wrong agent must read as "not found," identical to an unknown session_id.
    """

    async def create(self, agent: str) -> str:
        """Allocate a new session under `agent`, with empty history, and return its id."""
        ...

    async def get(self, agent: str, session_id: str) -> list[Message]:
        """Return the stored history for `(agent, session_id)`.

        Raises:
            SessionNotFoundError: if no session exists for this exact `(agent, session_id)`
                pair -- covers both an unknown id and an id that exists under another agent.
        """
        ...

    async def append(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Extend the stored history for `(agent, session_id)` with `messages`, in order.

        Raises:
            SessionNotFoundError: same condition as `get`.
        """
        ...

    async def replace(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Overwrite the stored history for `(agent, session_id)` with `messages` entirely.

        Raises:
            SessionNotFoundError: same condition as `get`.
        """
        ...

    def lock(self, agent: str, session_id: str) -> AbstractAsyncContextManager[None]:
        """Serialize operations against `(agent, session_id)` across concurrent requests.

        Used to guard any read-modify-write sequence spanning more than one call to this
        store (e.g. `CompactionService.compact()`'s get-summarize-replace) against a
        concurrent `append()` landing in between and being silently lost. Correctness is
        backend-specific: an in-process implementation can use a plain lock, but a durable
        or distributed implementation must provide a lock that's actually correct for its
        own backend (a distributed lock, a row lock) -- there is no universal way to
        implement this once at the protocol level.
        """
        ...

    def busy(self, agent: str, session_id: str) -> AbstractAsyncContextManager[None]:
        """Marks `(agent, session_id)` as in-flight for the context's duration.

        Raises SessionBusyError immediately if another operation already holds it for
        this same pair -- never waits. Distinct from `lock()`: `lock()` protects a brief
        store-mutation critical section (an implementation detail internal to one
        method); `busy()` protects an entire caller-defined operation (e.g. the whole
        `AgentRunService.run()` body, from before history is even read to after the
        final append) from a second concurrent operation on the same session, regardless
        of what either does internally. Two independent requests against the same
        session, run concurrently, would each read the same starting history and answer
        without the other's exchange -- silently confusing, not corrupting -- rejecting
        the second outright avoids that entirely.

        Raises:
            SessionBusyError: if another operation already holds `(agent, session_id)`.
        """
        ...

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and all its stored history.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        ...
