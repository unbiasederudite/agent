# protocols

Interfaces for anything with interchangeable implementations.

## Contents

- `illm.py` — `ILLM`, implemented by any outbound adapter that can turn messages into a completion (currently: `LiteLLMAdapter`). `complete()` takes optional `temperature`/`top_p`/`max_tokens`; `None` means "use the implementation's own configured default."
- `itool.py` — `ITool`, implemented by any concrete tool in `core/tools/` (currently: `GetCurrentTimeTool`). Declares `execute()`, but nothing calls it yet — `tool_calls` an LLM returns are passed through to the client unexecuted.
