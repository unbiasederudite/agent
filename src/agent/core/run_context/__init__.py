"""Per-run correlation context: the (agent, session_id) pair a run executes under."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple


class RunContext(NamedTuple):
    """The (agent, session_id) pair one run executes under."""

    agent: str  # Agent the run executes under.
    session_id: str | None  # Session the run executes under, or `None` for a new one.


_run_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def current_run_context() -> RunContext | None:
    """Return the current run's (agent, session_id), or None outside a run.

    Returns:
        RunContext | None: the current run context, or `None`.
    """
    return _run_context.get()


@contextmanager
def run_context(agent: str, session_id: str | None) -> Iterator[None]:
    """Bind `(agent, session_id)` as the current run context for this block.

    Args:
        agent: Agent the run executes under.
        session_id: Session the run executes under, or `None` for a new one.
    """
    token = _run_context.set(RunContext(agent, session_id))
    try:
        yield
    finally:
        _run_context.reset(token)


def update_session_id(session_id: str) -> None:
    """Update the current run context's session_id.

    Args:
        session_id: The newly created session's id.
    """
    current = _run_context.get()
    if current is not None:
        _run_context.set(RunContext(current.agent, session_id))
