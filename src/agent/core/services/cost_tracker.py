"""CostTracker: cumulative token/cost tracking per session and per agent."""

from collections import OrderedDict

from agent.core.models.usage import ZERO_USAGE, Usage, sum_usage


class CostTracker:
    """Tracks cumulative token/cost usage per session and per agent.

    A counter, not a gauge: every `record()` call adds to the running total, it never
    overwrites. A session's current context-token footprint (which goes up and down, not a running
    total) is a different kind of value entirely and lives in `ContextFootprintTracker`,
    not here -- see that class's own docstring for why.

    Owns its own private, non-durable state -- lost on restart, same posture every other
    per-session structure in this codebase already has.

    Two deliberate, currently-undocumented-elsewhere gaps in what gets reported: (a)
    `CompactionService`'s own summarizer LLM calls are real, billable completions, but
    their `Completion.usage` is discarded (only the summary text is kept), so their cost
    is never recorded here -- reported cost is a strict underestimate whenever compaction
    is enabled. (b) A run that fails partway through (a timeout, exhausted compaction
    retries, etc.) is never recorded at all, even though every LLM call it made before
    failing was real spend. Both are scope decisions from the design spec, not bugs.
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        """Initialize CostTracker.

        Args:
            max_sessions: Caps how many (agent, session_id) entries the cumulative
                session-usage state keeps at once, LRU-evicted -- same value, same
                eviction policy as every other per-session structure in this codebase.
                `None` means unbounded. Per-agent totals are never bounded: agent names
                come from the small, finite, config-declared registry, not user input.
        """
        self._max_sessions = max_sessions
        self._session_usage: OrderedDict[tuple[str, str], Usage] = OrderedDict()
        self._agent_usage: dict[str, Usage] = {}

    def record(self, agent: str, session_id: str, turn_usage: Usage) -> None:
        """Fold one run's usage into both the session's and the agent's cumulative totals."""
        key = (agent, session_id)
        self._session_usage[key] = sum_usage(self._session_usage.get(key, ZERO_USAGE), turn_usage)
        self._session_usage.move_to_end(key)
        if self._max_sessions is not None and len(self._session_usage) > self._max_sessions:
            self._session_usage.popitem(last=False)

        self._agent_usage[agent] = sum_usage(self._agent_usage.get(agent, ZERO_USAGE), turn_usage)

    def session_usage(self, agent: str, session_id: str) -> Usage | None:
        """Cumulative usage for this session.

        None if never recorded, including if it was recorded once but has since been
        LRU-evicted.
        """
        return self._session_usage.get((agent, session_id))

    def agent_usage(self, agent: str) -> Usage:
        """Cumulative usage for this agent, across every session. All-zero if never run."""
        return self._agent_usage.get(agent, ZERO_USAGE)

    def forget(self, agent: str, session_id: str) -> None:
        """Discard this session's cumulative-usage entry. No-op if never recorded.

        Does not touch the agent's own lifetime cumulative total -- that reflects
        everything ever spent under this agent, deletion of one session doesn't undo it.
        """
        self._session_usage.pop((agent, session_id), None)
