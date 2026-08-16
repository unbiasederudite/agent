# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_llm_registry()`, builds an `LLMRegistry` from an `AppConfig` JSON file
