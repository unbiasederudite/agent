# adapters

Outbound adapters — the clients that talk to external systems. Each implements a `core/protocols/` interface and translates third-party exceptions into `AgentError` subclasses at the boundary.

## Contents

- `litellm.py` — `LiteLLMAdapter`, implements `ILLM` via `litellm`, giving access to any provider litellm supports through one adapter. Constructed with per-model `temperature`/`top_p`/`max_tokens` defaults; each `complete()` call can override any of them.
