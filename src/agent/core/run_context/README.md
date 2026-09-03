# run_context

Cross-cutting per-run correlation context: the `(agent, session_id)` pair one run executes under.

## Contents

- `__init__.py` — `run_context()`, `current_run_context()`, and `update_session_id()`, a `ContextVar`-backed correlation context.
