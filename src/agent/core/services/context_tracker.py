"""ContextFootprintTracker: the single source of truth for a session's current context size."""

from collections import OrderedDict


class ContextFootprintTracker:
    """Tracks each session's current context-token footprint."""

    def __init__(self, max_sessions: int | None = None) -> None:
        """Initialize with no recorded footprints.

        Args:
            max_sessions: Cap on how many (agent, session_id) entries are kept,
                LRU-evicted; `None` means unbounded.
        """
        self._max_sessions = max_sessions
        self._footprints: OrderedDict[tuple[str, str], int] = OrderedDict()

    def record(self, agent: str, session_id: str, context_tokens: int) -> None:
        """Record this turn's ending context size for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to record for.
            context_tokens: The turn's ending context size, in tokens.
        """
        key = (agent, session_id)
        self._footprints[key] = context_tokens
        self._footprints.move_to_end(key)
        if self._max_sessions is not None and len(self._footprints) > self._max_sessions:
            self._footprints.popitem(last=False)

    def get(self, agent: str, session_id: str) -> int | None:
        """Return the context-token footprint last recorded for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to look up.

        Returns:
            int | None: the token count, or `None` if unknown.
        """
        return self._footprints.get((agent, session_id))

    def forget(self, agent: str, session_id: str) -> None:
        """Discard the recorded footprint for (agent, session_id). No-op if never recorded.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to forget.
        """
        self._footprints.pop((agent, session_id), None)
