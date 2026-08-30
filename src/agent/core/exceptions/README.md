# exceptions

The full exception hierarchy, rooted at `AgentError`. `core/` raises only `AgentError` subclasses; adapters catch third-party exceptions at the boundary and re-raise as one of these.

## Contents

- `__init__.py` — `AgentError` (the root), `ConfigError`, `LLMNotFoundError`, `AgentNotFoundError`, `ToolNotFoundError`, `StrategyNotFoundError`, `SessionNotFoundError`, `InputTooLargeError`, `LLMError` (with `LLMRateLimitedError`, `LLMTimeoutError`, `LLMContextWindowExceededError`, and its own subclass `CompactionExhaustedError` — raised only when `AgentRunService`'s compaction-and-retry attempt is exhausted, distinguishing "compaction was tried and could not help" from a plain overflow where compaction was never available to try)
