# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_registries()`, builds an `LLMRegistry`, `AgentRegistry`, and `ToolRegistry` from an `AppConfig` JSON file, cross-validating that every agent's `default_llm` is a declared LLM and every agent's `tools` entry is a declared tool with a known code-level implementation
