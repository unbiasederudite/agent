# run_context

Cross-cutting per-run correlation context: the `(agent, session_id)` pair one
`AgentRunService.run()` call executes under, threaded through logging the same way
`api/request_context.py`'s request id is.

## Contents

- `__init__.py` — `run_context(agent, session_id)`, a `ContextVar`-backed context manager
  bound around one `run()` call; `current_run_context()`, the read side; `update_session_id()`,
  called once a brand-new session is created (mid-run, after the id is known). Owned here,
  not in `api/`, because `core/` is what knows the agent and session_id in the first place --
  `api/logging_setup.py`'s `RunContextFilter` imports `current_run_context()` to stamp every
  `LogRecord` with it, the same way `RequestIdFilter` stamps the request id. `core/` never
  imports anything from `api/`, so the dependency only ever points that one direction.
