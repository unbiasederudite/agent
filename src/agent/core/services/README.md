# services

Use-case orchestration, called by inbound adapters.

## Contents

- `agent_run.py` — `AgentRunService`, the entry point for one agent turn: resolves config, runs the strategy, manages compaction and session history.
- `compaction.py` — `CompactionService`, keeps a session's stored history under its configured token budget.
- `context_tracker.py` — `ContextFootprintTracker`, tracks each session's current context-token footprint.
- `cost_tracker.py` — `CostTracker`, tracks cumulative token/cost usage per session and per agent.
- `session_service.py` — `SessionService`, session-lifecycle operations: reading history and usage, deleting a session.
