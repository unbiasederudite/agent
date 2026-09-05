"""CostTracker: cumulative token/cost tracking per session and per agent."""

from collections import OrderedDict

from agent.core.models.usage import ZERO_USAGE, Usage, sum_usage


class CostTracker:
    """Tracks cumulative token/cost usage per session and per agent."""

    def __init__(self, max_sessions: int | None = None) -> None:
        """Initialize with no recorded usage.

        Args:
            max_sessions: Cap on how many (agent, session_id) entries are kept,
                LRU-evicted; `None` means unbounded.
        """
        self._max_sessions = max_sessions
        self._session_usage: OrderedDict[tuple[str, str], Usage] = OrderedDict()
        self._agent_usage: dict[str, Usage] = {}

    def record(self, agent: str, session_id: str, turn_usage: Usage) -> None:
        """Fold one run's usage into both the session's and the agent's cumulative totals.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to record for.
            turn_usage: This run's usage.
        """
        key = (agent, session_id)
        self._session_usage[key] = sum_usage(self._session_usage.get(key, ZERO_USAGE), turn_usage)
        self._session_usage.move_to_end(key)
        if self._max_sessions is not None and len(self._session_usage) > self._max_sessions:
            self._session_usage.popitem(last=False)

        self._agent_usage[agent] = sum_usage(self._agent_usage.get(agent, ZERO_USAGE), turn_usage)

    def session_usage(self, agent: str, session_id: str) -> Usage | None:
        """Cumulative usage for this session.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to look up.

        Returns:
            Usage | None: the cumulative usage, or `None` if unknown.
        """
        return self._session_usage.get((agent, session_id))

    def agent_usage(self, agent: str) -> Usage:
        """Cumulative usage for this agent, across every session. All-zero if never run.

        Args:
            agent: Agent to look up.

        Returns:
            Usage: the agent's cumulative usage.
        """
        return self._agent_usage.get(agent, ZERO_USAGE)

    def forget(self, agent: str, session_id: str) -> None:
        """Discard this session's cumulative-usage entry.

        Args:
            agent: Agent the session belongs to.
            session_id: Session to forget.
        """
        self._session_usage.pop((agent, session_id), None)
