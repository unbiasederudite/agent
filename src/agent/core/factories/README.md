# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_registries()`, builds an `LLMRegistry` and `AgentRegistry` from an `AppConfig` JSON file, cross-validating that every agent's `default_llm` is a declared LLM
