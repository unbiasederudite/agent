"""Protocol interface for per-conversation message history storage."""

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
