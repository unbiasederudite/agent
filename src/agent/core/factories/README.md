# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_registries()`, builds the LLM, agent, tool, and strategy registries plus process-wide config from an already-parsed `AppConfig`.
