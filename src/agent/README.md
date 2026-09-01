# agent

Root package. See `ARCHITECTURE.md` for the full layering rationale.

## Contents

- `core/` — all agent intelligence; owns no I/O of its own. See `core/README.md`.
- `api/` — inbound HTTP adapter. See `api/README.md`.
- `adapters/` — outbound adapters to external systems (LLM providers, etc). See `adapters/README.md`.
