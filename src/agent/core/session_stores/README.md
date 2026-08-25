# session_stores

`ISessionStore` (`core/protocols/isession_store.py`) implementations: per-conversation message
history storage, keyed by `(agent, session_id)`.

## Contents

- `in_memory.py` — `InMemorySessionStore`, backed by a process-local dict. No durability (lost on
  restart), no eviction, no locking (single-threaded `asyncio`, no `await` mid-mutation). The only
  implementation this milestone; a durable backend (Redis/SQLite) can be added later behind the
  same protocol without touching any caller.
