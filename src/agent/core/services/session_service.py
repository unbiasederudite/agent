"""Session-lifecycle use cases spanning ISessionStore, CostTracker, and ContextFootprintTracker."""

import logging

from agent.core.models.message import Message
from agent.core.models.usage import ZERO_USAGE, Usage
from agent.core.protocols.isession_store import ISessionStore
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.services.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class SessionService:
    """Orchestrates session-lifecycle operations an inbound adapter shouldn't have to.

    A thin adapter (`api/`, and eventually `cli/`) translates its own request shape into
    a call here and the result back into its own response shape -- it never itself
    sequences `ISessionStore`, `CostTracker`, and `ContextFootprintTracker` together, so
    a second adapter never has to rediscover (and risk getting wrong) that deleting a
    session also means forgetting its recorded usage totals and context footprint.

    Has no direct dependency on `CompactionService` -- that service reads
    `ContextFootprintTracker` for its own purposes but keeps no state of its own for this
    service to forget on delete.
    """

    def __init__(
        self,
        session_store: ISessionStore,
        cost_tracker: CostTracker | None = None,
        context_tracker: ContextFootprintTracker | None = None,
    ) -> None:
        """Wrap `session_store` and the optional `cost_tracker`/`context_tracker`.

        Both default to a private, disposable instance when omitted -- see
        `AgentRunService.__init__`'s identical rationale. Every real caller
        (`create_app()`) always passes the same shared instances also given to
        `AgentRunService` (and, for `context_tracker`, to `CompactionService` too).
        """
        self._session_store = session_store
        self._cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self._context_tracker = (
            context_tracker if context_tracker is not None else ContextFootprintTracker()
        )

    async def get_history(self, agent: str, session_id: str) -> list[Message]:
        """Return the full stored history for `(agent, session_id)`.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        return await self._session_store.get(agent, session_id)

    async def get_usage(self, agent: str, session_id: str) -> tuple[Usage, int]:
        """Return (cumulative usage, context_tokens) for this session.

        Composes two independent reads after confirming the session exists: cumulative
        usage from `CostTracker` (an all-zero `Usage` if none was ever recorded -- e.g.
        the session exists but its `CostTracker` entry was LRU-evicted, reachable when
        `max_sessions` is configured, since `CostTracker`'s eviction, unlike
        `InMemorySessionStore`'s, doesn't skip a currently busy/locked session) and
        `context_tokens` from `ContextFootprintTracker` (`0` if unknown -- LRU-evicted, or
        reset after a successful compaction). Neither missing value is treated as an
        error; both default to their zero value independently.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
        """
        await self._session_store.get(agent, session_id)
        usage = self._cost_tracker.session_usage(agent, session_id)
        context_tokens = self._context_tracker.get(agent, session_id)
        return (
            usage if usage is not None else ZERO_USAGE,
            context_tokens if context_tokens is not None else 0,
        )

    async def delete(self, agent: str, session_id: str) -> None:
        """Permanently remove `(agent, session_id)` and forget its recorded usage state.

        Held under `ISessionStore.busy()` so a delete never races an in-flight
        `AgentRunService.run()` call against the same session. `CostTracker.forget()` and
        `ContextFootprintTracker.forget()` (each a no-op if there was nothing recorded)
        run only after the delete succeeds, so a session left mid-delete by a raised
        `SessionNotFoundError` never has its state forgotten for nothing.

        Raises:
            SessionNotFoundError: if no session exists for this exact pair.
            SessionBusyError: if another operation already holds this session.
        """
        async with self._session_store.busy(agent, session_id):
            await self._session_store.delete(agent, session_id)
            self._cost_tracker.forget(agent, session_id)
            self._context_tracker.forget(agent, session_id)
        logger.info("session deleted: agent=%s session=%s", agent, session_id)
