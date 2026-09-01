# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_registries()`, builds an `LLMRegistry`, `AgentRegistry`, `ToolRegistry`, `StrategyRegistry`, the process-wide `base_prompt`, the raw `CompactionConfig` (if any), and the raw `LoggingConfig` from an already-parsed `AppConfig` (reading/parsing the JSON file is I/O, so that's `api/app.py`'s `create_app()`'s job, not this factory's -- `core/` owns none), one `_build_*_registry()` helper per registry, each following the same shape: declared config entries resolve against a code-level implementation map, raising `ConfigError` on a duplicate name or missing implementation. Cross-validates that every agent's `model` is a declared LLM, `strategy` is a declared strategy, every `tools` entry is a declared tool with a known code-level implementation, and `compaction.model` (if set) is a declared LLM. Does not itself configure logging -- `api/app.py`'s `create_app()` builds that from the returned `LoggingConfig`, since logging setup needs things (a request-id filter, whether this is even an HTTP context) not owned by this factory
