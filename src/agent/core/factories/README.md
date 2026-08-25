# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_registries()`, builds an `LLMRegistry`, `AgentRegistry`, `ToolRegistry`, `StrategyRegistry`, and the process-wide `base_prompt` from an `AppConfig` JSON file, one `_build_*_registry()` helper per registry, each following the same shape: declared config entries resolve against a code-level implementation map, raising `ConfigError` on a duplicate name or missing implementation. Cross-validates that every agent's `model` is a declared LLM, `strategy` is a declared strategy, and every `tools` entry is a declared tool with a known code-level implementation
