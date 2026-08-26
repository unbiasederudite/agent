# core

All agent intelligence. Owns no I/O of its own — `api/` and `adapters/` are thin translators at the edges.

## Contents

- `models/` — Pydantic domain and startup-config models
- `protocols/` — interfaces for interchangeable implementations
- `registries/` — runtime name-to-instance maps
- `factories/` — construct registered instances from config
- `services/` — use-case orchestration
- `exceptions/` — the `AgentError` hierarchy
