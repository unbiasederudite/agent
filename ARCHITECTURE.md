# Architecture

## Folder Responsibility

| Folder | Purpose |
|---|---|
| `src/agent/cli/` | Inbound adapter — terminal interaction |
| `src/agent/api/` | Inbound adapter — HTTP interaction |
| `src/agent/core/` | All agent intelligence |
| `src/agent/core/exceptions/` | Full exception hierarchy rooted at `AgentError` |
| `src/agent/core/factories/` | Factories that construct registered instances from configuration |
| `src/agent/core/models/` | All Pydantic data models: domain models and startup config |
| `src/agent/core/protocols/` | Protocol interfaces for anything with interchangeable implementations |
| `src/agent/core/registries/` | Runtime registries for agents, LLMs, and tools |
| `src/agent/core/services/` | Use-case orchestration |
| `src/agent/core/strategies/` | Reasoning and selection algorithm implementations |
| `src/agent/core/tools/` | Concrete tool implementations |
| `src/agent/adapters/` | Outbound adapters — the clients that talk to external systems |
| `tests/unit/` | Pure logic tests — no external deps |
| `tests/integration/` | Adapter wiring tests — all externals mocked |
| `tests/e2e/` | Real-runtime smoke tests — never in CI |

`core/` is a deliberate boundary, not just another folder: it owns all agent intelligence. `cli/`, `api/`, and `adapters/` are thin — they translate at the edges and contain no reasoning of their own.

---

## Inbound vs Outbound Adapters

- **Inbound adapters** — `cli/` and `api/`. The outside world calls **into** the core through these. Each one's job: translate its own input into a call against `core/services/`, and translate the result back into its own output format.
- **Outbound adapters** — `adapters/`. The core calls **out** through these, to an LLM provider, a tool's REST client, a database.

Inbound adapters are siblings of each other, never nested inside one another, and `core/` never depends on any of them.

Each inbound adapter is a separate `[project.optional-dependencies]` extra with its own entry point. `pip install agent-core` installs zero adapter-specific dependencies — a deployment installs only the extra it needs.

---

## Dependency Flow

```
inbound adapter -> services -> strategies -> protocols -> outbound adapters
```

Cross-cutting (available to every layer): `models`, `exceptions`, `logging.Logger`

Adapters implement the protocol interfaces and therefore import from `core/protocols/`. The arrow shows the runtime call direction, not the import direction.

No circular imports. No layer calling backwards. No inbound adapter depends on another.

---

## Config System

Everything configurable is declared in a single JSON file loaded and validated at startup. Nothing can be changed after the process starts. The configurable surface is:

- **LLM registry** — available providers and their models
- **Agent registry** — available agents, each with a system prompt, a strategy, a default LLM, and tools
- **Tool registry** — available tools and their settings
- **Compaction** — the LLM and settings used for compaction, plus the token budget that triggers it
- **Logging** — minimum log level

---

## Registries and Factories

Registries and factories are the binding layer between config and runtime.

- **Config declares names.** Each registry section in the config lists what is available by name.
- **Factories resolve names to instances.** `core/factories/` reads the config and constructs concrete instances at startup. An inbound adapter calls into it — it never does this wiring itself.
- **Registries answer queries.** At runtime, the core requests an instance by name. If the name is not registered, a startup error is raised immediately — not at invocation time.
- **Absent names are inactive, not errors.** A name present in code but absent from config is silently unavailable. A name declared in config but not wired by a factory fails fast at startup.
- **Wiring happens once per process.** The factory builds everything from config at startup; nothing re-reads config afterward.

---

## Wire Format

The internal wire format for LLM messages and tool-calling is OpenAI-compatible. Any LLM adapter that speaks the OpenAI format drops in without touching `core/`, and `api/` can expose an OpenAI-compatible endpoint itself, enabling drop-in use by existing OpenAI clients.

---

## Conversation Model

The platform supports one active conversation at a time. Session history is held in memory only — it is not persisted and is discarded when the conversation ends. A new conversation always starts from a clean state.

Within a conversation, each user request drives dynamic selection of which agent, LLM, and tools to activate. The config declares the available universe; the request determines what is used.

---

## Exception Hierarchy

All exceptions are subclasses of `AgentError`. `core/` raises only `AgentError` subclasses. Adapters catch external exceptions and re-raise as the appropriate subclass — raw third-party exceptions must never propagate past the adapter boundary.

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
| `XConfig` | Pydantic config model in `core/models/` |
| `XError` | Typed exception in `core/exceptions/` |
| `logging.getLogger(__name__)` | Standard logger (one per module) |
| Plain noun | Data-only `BaseModel` |
