# session_stores

`ISessionStore` (`core/protocols/isession_store.py`) implementations: per-conversation message
history storage, keyed by `(agent, session_id)`.

## Contents

- `in_memory.py` — `InMemorySessionStore`, an in-process, non-durable `ISessionStore` backed by a dict, bounded by `max_sessions` via LRU eviction.
