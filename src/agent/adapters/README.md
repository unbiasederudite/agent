# adapters

Outbound adapters — the clients that talk to external systems. Each implements a `core/protocols/` interface and translates third-party exceptions into `AgentError` subclasses at the boundary.

## Contents

- `litellm.py` — `LiteLLMAdapter`, the `ILLM` implementation backed by `litellm`, giving access to any provider litellm supports.
