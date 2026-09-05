# Architecture

This project runs AI agents defined entirely by config: config declares the process-wide set
of available LLMs, tools, reasoning strategies, and guardrails, plus process behavior like
compaction, logging, and session storage; each agent picks which of those it uses, and is
exposed to callers through a transport. Every one of those pieces is swappable — a new backend
means implementing the matching protocol, not touching the code that orchestrates a run.
`agent/core` is what makes that possible: it separates all agent reasoning from the transports
that expose it and the concrete backends it calls out to. This document describes that
separation and the invariants that hold it together — read it before any change that touches
more than one folder.

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
| `src/agent/core/registries/` | Runtime registries for agents, LLMs, tools, strategies, and guardrails |
| `src/agent/core/run_context/` | Per-run `(agent, session_id)` correlation context, threaded through logging, plus a per-run accumulator for supporting-LLM-call usage |
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
- **This runs as a single process.** Every registry, the session store, cost/context
  tracking, and `core/run_context/`'s per-run state are in-process objects with no
  cross-process coordination. Running more than one instance (multiple workers, multiple
  replicas behind a load balancer) gives each its own independent state, not a shared view —
  a session created on one instance doesn't exist on another.

```
inbound adapter -> services -> strategies -> protocols -> outbound adapters
```

Cross-cutting (available to every layer): `models`, `exceptions`, `logging.Logger`,
`core/run_context/`. The arrow above is the runtime call direction, not the import
direction — adapters import from `core/protocols/` to implement it. No circular imports; no
layer calls backwards.

---

## Config System

Everything configurable is declared in a single JSON file, loaded and validated once at
startup; nothing changes after the process starts. The configurable surface: the LLM, agent,
strategy, tool, and guardrail registries; the session store; compaction (summarizer model,
budget, keep-window); and logging (level, format, destinations).

Registries and factories are the binding layer between that config and runtime:

- **Config declares names**; **factories** (`core/factories/`) resolve those names to
  concrete instances once at startup; **registries** answer runtime lookups by name and
  raise immediately if a name isn't registered — never at first use.
- A name present in code but absent from config is silently unavailable. A name in config
  with no matching implementation fails fast at startup, not at invocation time.
- **An agent's `model`, `strategy`, and `tools` are defaults, not fixed choices.** A request
  may override any of them; each has a matching ceiling (`allowed_models`,
  `allowed_strategies`, `allowed_tools`) bounding what a request may override *to* — `None`
  means unrestricted, and a default outside its own ceiling fails validation at config-load
  time. Guardrail lists (`input_guardrails`, `tool_output_guardrails`, `output_guardrails`)
  have no such override: they're fixed per agent, never chosen per request.
- A component with no config surface of its own (nothing per-instance to resolve, e.g.
  logging setup) is constructed directly by its caller instead of routed through
  `core/factories/`. Follow this precedent rather than growing a factory for a component
  with nothing to configure.

---

## Extending

Adding a new backend for a config-driven protocol (`ILLM`, `IStrategy`, `ITool`, `IGuardrail`,
`ISessionStore`) means implementing it and registering the implementation in
`core/factories/app.py`; no other file in `core/` needs to change.

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
- **Cost and context tracking are independent** of each other and of compaction. Cumulative
  cost *is* recorded for a run's supporting LLM calls — compaction's summarizer, a guardrail's
  own internal LLM check — not just its main turn. The per-request `Run` keeps them apart
  instead of folding them together: `usage` is the turn alone, `supporting_usage` is
  everything else, so `usage` stays a reliable size signal for the conversation while the
  total cost is still `usage + supporting_usage`. Cost isn't recorded for a run that fails
  partway through, nor for compaction triggered outside a run, so reported cost is still a
  floor, not an exact total.
- **Per-run growth caps**, optional per agent, bound one in-progress run independent of
  compaction: a cap on one tool result's length, on tool calls executed per round, and on
  combined tool-result content across the whole run. Both the per-round and the aggregate cap
  are visible to the LLM rather than silent: a tool call beyond the per-round cap becomes an
  explicit skipped-call error message, and content beyond the aggregate cap is replaced with
  an omission marker — the same "let it react" contract guardrails follow.
- **Tool calls within one round execute concurrently** (`asyncio.gather`), not in sequence.
  If the loop's `max_tool_iterations` is exhausted before the model stops requesting tools,
  one final call is forced with no tools offered, so the loop always terminates in an answer.
- **A run's timeout doesn't roll back other side effects.** `max_request_seconds` bounds the
  whole call via `asyncio.wait_for`; if a proactive compaction already committed its rewrite
  to the session store before the timeout fires, that rewrite stands — the run still raises
  `RequestTimeoutError`, but the session's stored history has already changed.

---

## Guardrails

A guardrail is a named reference to a dynamically-resolved [Guardrails AI](https://guardrailsai.com/)
Hub validator — this codebase ships zero built-in checks. Resolution first tries the current,
supported path: deriving the validator's PyPI-published Python package from its Hub id
(`namespace/name` → `guardrails_ai.name`) and importing it. If that package doesn't exist —
true for validators not yet republished under that convention — Guardrails AI's own validator
lookup transparently falls back to the older, deprecated Hub CLI/registry mechanism, so a
validator only ever installed that way keeps working. Either way, resolution ends by looking
the class up by its original Hub id in Guardrails AI's own registry, not by import path. An
unresolvable `validator_id` fails at startup as a `ConfigError`, not at first use.

Three checkpoints, independently configured per agent: the incoming message
(`input_guardrails`), each tool result (`tool_output_guardrails`), and the final response
(`output_guardrails`). Each guardrail carries its own action — `block` (raise
`GuardrailBlockedError`), `redact` (substitute corrected content and continue), or `warn` (log
and continue unchanged). If the underlying validator itself raises, that never propagates as an
exception: `block` treats the failure as a genuine trigger (a check that couldn't run has
verified nothing); `warn` and `redact` just log the failure and let content through unchanged,
since neither has a corrected value to substitute when the check never completed.

The tool-output checkpoint is the one asymmetric case: a block there never raises — it becomes
a tool-result error message the LLM can react to, rather than aborting the run. Input and
output blocks do raise `GuardrailBlockedError`.

Some Hub validators call an LLM internally via a constructor argument named `llm_callable`.
When a resolved validator declares that parameter, `validator_params.llm_callable` must name a
model registered in this process's `llms` config; that call is then transparently redirected
through the matching `ILLM` instead of a provider, sharing its retry/timeout behavior with every
other completion. Detection is by constructor introspection, so it only sees `llm_callable` as
an explicit, named parameter — a validator that only accepts it via `**kwargs` is invisible to
it the same as one with no such parameter at all. The redirect also patches litellm's own
provider-resolution utility, not just its completion call: litellm's completion function
consults its custom-provider registrations itself, but that separate resolution utility doesn't
— and some Hub validators call the utility directly before completing — so left unpatched, any
redirected call reaching one of those validators would fail before it was ever made.

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
