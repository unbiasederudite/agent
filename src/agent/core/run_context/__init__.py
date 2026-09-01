"""Per-run correlation context: the (agent, session_id) pair a run executes under.

Threaded through logging the same way `api/request_context.py`'s request_id is: a
`ContextVar` set once per run, read by a `logging.Filter` attached to every handler --
`core/` code that logs during a run needs no awareness of this module at all. Defined
here, not in `api/`, because `core/` is what knows the agent and session_id in the
first place; `api/logging_setup.py` imports `current_run_context()` to build that
Filter -- `core/` never imports anything from `api/`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple


class RunContext(NamedTuple):
    """The (agent, session_id) pair one run executes under."""

    agent: str
    session_id: str | None


_run_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def current_run_context() -> RunContext | None:
    """Return the current run's (agent, session_id), or None outside a run."""
    return _run_context.get()


@contextmanager
def run_context(agent: str, session_id: str | None) -> Iterator[None]:
    """Bind (agent, session_id) to every log line emitted for the rest of this block.

    `session_id` is None for a brand-new session until `update_session_id` is called
    once one is created.
    """
    token = _run_context.set(RunContext(agent, session_id))
    try:
        yield
    finally:
        _run_context.reset(token)


def update_session_id(session_id: str) -> None:
    """Update the current run's session_id once a brand-new session has been created.

    No-op outside a `run_context` block (defensive; should never happen in practice).
    """
    current = _run_context.get()
    if current is not None:
        _run_context.set(RunContext(current.agent, session_id))
