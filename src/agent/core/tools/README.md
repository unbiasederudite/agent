# tools

Concrete tool implementations, exposed to an LLM as function-calling schemas via
`ToolConfig`/`ToolRegistry`. `ITool.execute()` is declared on the interface but not yet
called anywhere -- `tool_calls` an LLM returns are passed through to the client unexecuted
until a future milestone adds `core/strategies/` to own that loop.

## Contents

- `get_current_time.py` -- `GetCurrentTimeTool`, returns the current UTC time as an ISO
  8601 string. No parameters, no network, no config; exists to prove the tool-calling
  plumbing end-to-end.
