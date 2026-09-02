"""ContextFootprintTracker: the single source of truth for a session's current context size."""

from collections import OrderedDict


class ContextFootprintTracker:
    """Tracks each session's current context-token footprint -- a gauge, not a counter.

    Unlike cumulative usage (`CostTracker`, which only ever sums), this value goes up
    and down: it's "how big is this session's stored history right now," overwritten
    every turn, and it can shrink (e.g. after a successful compaction).

    Written only by `AgentRunService`, once per successful run. Read by `CompactionService`
    (to decide whether a session is over budget) and by the usage API (via `SessionService`,
    for the `context_tokens` field) -- neither of those two depends on the other; both
    depend only on this. Putting the value on either of them instead would either create a
    circular dependency (`CompactionService` already depends on `AgentRunService`'s caller
    chain) or make the usage endpoint's `context_tokens` disappear whenever compaction is
    disabled (`CompactionService` isn't constructed at all in that case). A third,
    dependency-free component avoids both problems.

    Owns its own private, non-durable state -- lost on restart, same posture every other
    per-session structure in this codebase already has.
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        """Initialize ContextFootprintTracker.

        Args:
            max_sessions: Caps how many (agent, session_id) entries are kept at once,
                LRU-evicted -- same value, same eviction policy as every other per-session
                structure in this codebase. `None` means unbounded. Eviction does not skip
                a currently busy/locked session (unlike `InMemorySessionStore`'s own
                eviction) -- a session can outlive its recorded footprint under sustained
                eviction pressure; callers must treat a missing entry as "unknown," never
                as "zero."
        """
        self._max_sessions = max_sessions
        self._footprints: OrderedDict[tuple[str, str], int] = OrderedDict()

    def record(self, agent: str, session_id: str, context_tokens: int) -> None:
        """Record this turn's ending context size.

        Overwrites whatever was recorded last for this session -- a point-in-time
        snapshot of "how big is the stored history right now," not a running sum.
        """
        key = (agent, session_id)
        self._footprints[key] = context_tokens
        self._footprints.move_to_end(key)
        if self._max_sessions is not None and len(self._footprints) > self._max_sessions:
            self._footprints.popitem(last=False)

    def get(self, agent: str, session_id: str) -> int | None:
        """Return the context-token footprint last recorded for this session, or None.

        None if nothing has ever been recorded, or if it was recorded once but has since
        been LRU-evicted, or was explicitly forgotten (e.g. after a successful compaction,
        whose whole point is that the old footprint number no longer describes the
        session's real size).
        """
        return self._footprints.get((agent, session_id))

    def forget(self, agent: str, session_id: str) -> None:
        """Discard the recorded footprint for (agent, session_id). No-op if never recorded."""
        self._footprints.pop((agent, session_id), None)
