# tools

Concrete tool implementations, exposed to an LLM as function-calling schemas via
`ToolConfig`/`ToolRegistry`. `ITool.execute()` is invoked by whichever `IStrategy` a
request routes through (currently `ReactStrategy`, `core/strategies/`).

## Contents

- `get_current_time.py` — `GetCurrentTimeTool`, returns the current time, optionally at a UTC offset.
