# services

Use-case orchestration, called by inbound adapters. Each file's own docstrings carry the
full rationale for its design decisions -- this file is a short index, not a duplicate.

## Contents

- `agent_run.py` — `AgentRunService.run()`, the single entry point for one agent turn: resolves the agent/model/strategy/tools, enforces `max_input_chars` / allow-list ceilings / `max_request_seconds`, threads and (if configured) proactively compacts session history, delegates to the selected `IStrategy`, and reactively retries once through compaction on a context-window overflow. Binds `core.run_context.run_context()` around the whole call (see `core/run_context/README.md`) and rejects a concurrent second call on the same session via `ISessionStore.busy()`.
- `compaction.py` — `CompactionService`, keeps a session's stored history under its configured token budget: `record_usage`/`maybe_compact` for the proactive path, `compact` (called directly by `agent_run.py` on a reactive overflow) for the summarization itself, `forget` to discard a deleted session's usage estimate. Never corrupts stored history -- an unusable or non-shrinking summary leaves history untouched. Falls back to chunked map-reduce summarization if a single-pass summary itself overflows.
- `session_service.py` — `SessionService`, the session-lifecycle use case an inbound adapter shouldn't reimplement itself: `get_history()` is a passthrough, `delete()` orchestrates `ISessionStore.busy()` + `ISessionStore.delete()` + `CompactionService.forget()` together so a future `cli/` adapter reuses this instead of rediscovering the same sequence.
