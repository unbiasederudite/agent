# Architecture

agent/core separates all agent reasoning from the transports that expose it. This document
describes that separation and the invariants that hold it together — read it before any
change that touches more than one folder.

## Building Blocks

| Folder | Purpose |
|---|---|
| `src/agent/cli/` | Inbound adapter — terminal interaction (planned, not yet implemented) |
| `src/agent/api/` | Inbound adapter — HTTP interaction |
| `src/agent/core/` | All agent intelligence |
| `src/agent/core/exceptions/` | Full exception hierarchy rooted at `AgentError` |
| `src/agent/core/factories/` | Factories that construct registered instances from configuration |
| `src/agent/core/models/` | All Pydantic data models: domain models and startup config |
| `src/agent/core/protocols/` | Protocol interfaces for anything with interchangeable implementations |
| `src/agent/core/registries/` | Runtime registries for agents, LLMs, tools, and strategies |
| `src/agent/core/run_context/` | Per-run `(agent, session_id)` correlation context, threaded through logging |
| `src/agent/core/services/` | Use-case orchestration |
| `src/agent/core/session_stores/` | Per-conversation message history storage implementations |
| `src/agent/core/strategies/` | Reasoning and selection algorithm implementations |
| `src/agent/core/tools/` | Concrete tool implementations |
| `src/agent/adapters/` | Outbound adapters — the clients that talk to external systems |
| `tests/unit/` | Pure logic tests — no external deps |
| `tests/integration/` | Adapter wiring tests — all externals mocked |
| `tests/e2e/` | Real-runtime smoke tests — never in CI |

`core/` is a deliberate boundary, not just another folder: it owns all agent intelligence.
`cli/`, `api/`, and `adapters/` are thin — they translate at the edges and contain no
reasoning of their own.

---

## Boundaries

- **Inbound adapters** (`cli/`, `api/`) — the outside world calls **into** the core through
  these. Each translates its own input into a call against `core/services/`, and the result
  back into its own output format. They are siblings, never nested inside one another, and
  `core/` never depends on any of them.
- **Outbound adapters** (`adapters/`) — the core calls **out** through these, to an LLM
  provider, a tool's REST client, a database. They implement `core/protocols/` interfaces.
- Each inbound adapter is meant to ship as its own installable extra, so a deployment
  installs only the transport it needs — not yet true of the current `pyproject.toml`,
  where `api/`'s dependencies (FastAPI, uvicorn) are unconditional.

```
inbound adapter -> services -> strategies -> protocols -> outbound adapters
```

Cross-cutting (available to every layer): `models`, `exceptions`, `logging.Logger`. The
arrow above is the runtime call direction, not the import direction — adapters import from
`core/protocols/` to implement it. No circular imports; no layer calls backwards.

---

## Config System

Everything configurable is declared in a single JSON file, loaded and validated once at
startup; nothing changes after the process starts. The configurable surface: the LLM, agent,
tool, and strategy registries; compaction (summarizer model, budget, keep-window); and
logging (level, format, destinations).

Registries and factories are the binding layer between that config and runtime:

- **Config declares names**; **factories** (`core/factories/`) resolve those names to
  concrete instances once at startup; **registries** answer runtime lookups by name and
  raise immediately if a name isn't registered — never at first use.
- A name present in code but absent from config is silently unavailable. A name in config
  with no matching implementation fails fast at startup, not at invocation time.
- A component with no config surface of its own (nothing per-instance to resolve, e.g. the
  in-memory session store, or logging setup) is constructed directly by its caller instead
  of routed through `core/factories/`. Follow this precedent rather than growing a factory
  for a component with nothing to configure.

---

## Wire Format

The internal wire format for LLM messages and tool-calling is OpenAI-compatible.
`ILLM` implementations are contractually required to flatten tool-call/tool-result content
out of any outbound request that declares no `tools` — some providers reject tool-shaped
content on a request with no tool schema attached, even when it's just replaying history.
The leading system message is assembled from the optional root `base_prompt` followed by
the selected agent's own `system_prompt`, concatenated unconditionally and never overridden
by client-supplied messages.

---

## Conversation Model

Each conversation is a session, identified by a server-generated `session_id` and locked to
the agent it was created under. Session history lives only in memory (`ISessionStore`) — not
persisted, lost on restart; durability, not growth, is the store's known limitation.

- **Bounded by count, not time.** Sessions are never TTL-trimmed; per-session state (stored
  history, cumulative cost, context-token footprint) is capped via independent LRU eviction.
  A session can outlive its cost/footprint entry under eviction pressure — callers treat a
  missing entry as unknown, not zero.
- **Two concurrency primitives.** A read-modify-write spanning more than one store call is
  serialized via `lock()`; a whole second operation against the same session is rejected
  outright via `busy()`, distinct from `lock()`.
- **Compaction bounds cross-turn growth.** Once stored history crosses a configured token
  budget, its older portion is summarized and replaced, backed by a reactive retry (on an
  actual context-window overflow) and a chunked map-reduce fallback (when even the old
  portion overflows the summarizer). Compaction only touches already-stored history between
  turns — growth *inside* one in-progress run is bounded separately, below. A strategy's
  retry after an overflow reruns its whole loop from the start, so an already-executed tool
  call can run again; safe today only because the one existing tool is read-only.
- **Cost and context tracking are independent** of each other and of compaction, and neither
  is exhaustive: cost isn't recorded for compaction's own summarizer calls, nor for a run
  that fails partway through, so reported cost is a floor, not an exact total.
- **Per-run growth caps**, optional per agent, bound one in-progress run independent of
  compaction: a cap on one tool result's length, on tool calls executed per round, and on
  combined tool-result content across the whole run.

---

## Exception Hierarchy

All exceptions are subclasses of `AgentError`; `core/` raises only these, and adapters
re-raise third-party exceptions as the appropriate subclass at the boundary. `api/`'s request-id
middleware wraps the whole request in its own exception handler, ahead of Starlette's own
catch-all — so it, not the app's registered `Exception` handler, is what actually fires for a
genuinely unhandled exception during a request.

---

## Naming Conventions

| Pattern | Usage |
|---------|-------|
| `BaseX` | Abstract base class |
| `IX` | Interface / Protocol in `core/protocols/` |
| `XStrategy` | Interchangeable algorithm implementations in `core/strategies/` |
| `XService` | Orchestration and use-case logic in `core/services/` |
| `XRegistry` | Runtime name-to-instance maps in `core/registries/` |
| `XFactory` | Object construction from config in `core/factories/`, when a class earns its keep (state, multiple methods). A single construction step is a plain function instead. |
| `XAdapter` | Concrete outbound adapter for an external system in `adapters/` |
| `XSessionStore` | Interchangeable session-history storage implementations in `core/session_stores/` |
| `XConfig` | Pydantic config model in `core/models/` |
| `XError` | Typed exception in `core/exceptions/` |
| `logging.getLogger(__name__)` | Standard logger (one per module) |
| Plain noun | Data-only `BaseModel` |
