# exceptions

The full exception hierarchy, rooted at `AgentError`. `core/` raises only `AgentError` subclasses; adapters catch third-party exceptions at the boundary and re-raise as one of these.

## Contents

- `__init__.py` — `AgentError` (the root), `ConfigError`, `LLMNotFoundError`, `AgentNotFoundError`, `ToolNotFoundError`, `LLMError` (with `LLMRateLimitedError`, `LLMTimeoutError`)
