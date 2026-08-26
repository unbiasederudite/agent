"""In-process, non-durable ISessionStore implementation."""

import uuid

from agent.core.exceptions import SessionNotFoundError
from agent.core.models.message import Message


class InMemorySessionStore:
    """ISessionStore backed by a process-local dict. Lost on restart.

    No locking: every method body has no `await` before it finishes mutating state, so
    under asyncio's single-threaded cooperative scheduling no other coroutine can
    interleave mid-operation.
    """

    def __init__(self) -> None:
        """Initialize an empty session store."""
        self._sessions: dict[tuple[str, str], list[Message]] = {}

    async def create(self, agent: str) -> str:
        """Allocate a new session under `agent`, with empty history, and return its id."""
        session_id = uuid.uuid4().hex
        self._sessions[(agent, session_id)] = []
        return session_id

    async def get(self, agent: str, session_id: str) -> list[Message]:
        """Return the stored history for `(agent, session_id)`.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        key = (agent, session_id)
        if key not in self._sessions:
            raise SessionNotFoundError(f"no session '{session_id}' for agent '{agent}'")
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
