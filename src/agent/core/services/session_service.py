"""Session-lifecycle use cases spanning ISessionStore and CompactionService."""

import logging

from agent.core.models.message import Message
from agent.core.protocols.isession_store import ISessionStore
from agent.core.services.compaction import CompactionService

logger = logging.getLogger(__name__)


class SessionService:
    """Orchestrates session-lifecycle operations an inbound adapter shouldn't have to.

    A thin adapter (`api/`, and eventually `cli/`) translates its own request shape into
    a call here and the result back into its own response shape -- it never itself
    sequences `ISessionStore` and `CompactionService` together, so a second adapter never
    has to rediscover (and risk getting wrong) that deleting a session also means
    forgetting its recorded compaction usage estimate.
    """

    def __init__(
        self, session_store: ISessionStore, compaction_service: CompactionService | None
    ) -> None:
        """Wrap `session_store` and the optional `compaction_service` to orchestrate."""
        self._session_store = session_store
        self._compaction_service = compaction_service

    async def get_history(self, agent: str, session_id: str) -> list[Message]:
        """Return the full stored history for `(agent, session_id)`.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        return await self._session_store.get(agent, session_id)

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and forget its compaction estimate.

        Held under `ISessionStore.busy()` so a delete never races an in-flight
        `AgentRunService.run()` call against the same session. `CompactionService.forget()`
        (a no-op if compaction isn't configured, or the session was never estimated) runs
        only after the delete succeeds, so a session left mid-delete by a raised
        `SessionNotFoundError` never has its estimate forgotten for nothing.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
            SessionBusyError: if another operation already holds this session.
        """
        async with self._session_store.busy(agent, session_id):
            await self._session_store.delete(agent, session_id)
            if self._compaction_service is not None:
                self._compaction_service.forget(agent, session_id)
        logger.info("session deleted: agent=%s session=%s", agent, session_id)
