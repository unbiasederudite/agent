# protocols

Interfaces for anything with interchangeable implementations.

## Contents

- `illm.py` — `ILLM`, implemented by any outbound adapter that can turn messages into a completion (currently: `LiteLLMAdapter`). `complete()` takes optional `temperature`/`top_p`/`max_tokens`; `None` means "use the implementation's own configured default."
- `itool.py` — `ITool`, implemented by any concrete tool in `core/tools/` (currently: `GetCurrentTimeTool`). `execute()` is called by whichever `IStrategy` implementation the request routes through.
- `istrategy.py` — `IStrategy`, implemented by any reasoning-loop algorithm in `core/strategies/` (currently: `ReactStrategy`). Owns how (and whether) tools get offered to and invoked by the LLM; receives tools pre-resolved to instances (`dict[str, ITool]`) — name resolution against `ToolRegistry` is the caller's (`AgentRunService`'s) job, so a strategy never sees a tool it wasn't explicitly given. Returns a `Turn`, not a `Completion` — the aggregate of everything the run generated, not one wire call's result.
- `isession_store.py` — `ISessionStore`, implemented by any per-conversation history store in `core/session_stores/` (currently: `InMemorySessionStore`). Async even though the current implementation does no I/O, since a durable backend is the reason this is a protocol at all.
