# services

Use-case orchestration, called by inbound adapters.

## Contents

- `agent_run.py` — `AgentRunService`, resolves a required agent (and optional model/strategy/tools override) by name, builds the initial system+user messages (merging the process-wide `base_prompt`, if configured, into the agent's `system_prompt`), resolves tool names against `ToolRegistry` into instances before the strategy ever runs, threads a session's stored history (via `ISessionStore`) between the system message and the new user turn when `session_id` is given, and delegates to the selected `IStrategy` to run the reasoning loop and produce a `Run`
