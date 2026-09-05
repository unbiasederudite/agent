# adapters

Outbound adapters — the clients that talk to external systems. Each implements a `core/protocols/` interface and translates third-party exceptions into `AgentError` subclasses at the boundary.

## Contents

- `litellm.py` — `LiteLLMAdapter`, the `ILLM` implementation backed by `litellm`, giving access to any provider litellm supports.
- `guardrails_ai.py` — `GuardrailsAIAdapter`, the `IGuardrail` implementation backed by a dynamically-resolved Guardrails AI Hub validator, plus its `resolve_validator()` and `declares_llm_callable()` resolution helpers.
- `llm_registry_provider.py` — `LLMRegistryProvider` and `register_llm_registry_provider()`, a litellm custom-provider handler that redirects a Hub validator's own LLM calls through a registered `ILLM` instead of calling a provider directly, plus `guardrail_call_scope()`, which reattributes that call's usage.
