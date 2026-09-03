"""Protocol interface for per-conversation message history storage."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from agent.core.models.message import Message


class ISessionStore(Protocol):
    """Interface for per-conversation message history storage, keyed by (agent, session_id)."""

    async def create(self, agent: str) -> str:
        """Allocate a new session under `agent`, with empty history, and return its id.

        Args:
            agent: Agent the new session belongs to.

        Returns:
            str: the new session's id.
        """
        ...

    async def get(self, agent: str, session_id: str) -> list[Message]:
        """Return the stored history for `(agent, session_id)`.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to read.

        Returns:
            list[Message]: the session's stored history.

        Raises:
            SessionNotFoundError: if the session does not exist.
        """
        ...

    async def append(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Extend the stored history for `(agent, session_id)` with `messages`, in order.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to extend.
            messages: Messages to append.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        ...

    async def replace(self, agent: str, session_id: str, messages: list[Message]) -> None:
        """Overwrite the stored history for `(agent, session_id)` with `messages` entirely.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to overwrite.
            messages: Messages to store in place of the existing history.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        ...

    def lock(self, agent: str, session_id: str) -> AbstractAsyncContextManager[None]:
        """Serialize operations against `(agent, session_id)` for read-modify-write correctness.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to lock.
        """
        ...

    def busy(self, agent: str, session_id: str) -> AbstractAsyncContextManager[None]:
        """Mark `(agent, session_id)` as in-flight, rejecting immediately if already held.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to mark busy.

        Raises:
            SessionBusyError: if the session is already busy.
        """
        ...

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and all its stored history.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to delete.

        Raises:
            SessionNotFoundError: if the session does not exist.
        """
        ...
