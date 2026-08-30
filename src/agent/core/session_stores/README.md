# session_stores

`ISessionStore` (`core/protocols/isession_store.py`) implementations: per-conversation message
history storage, keyed by `(agent, session_id)`.

## Contents

- `in_memory.py` — `InMemorySessionStore`, backed by a process-local dict. No durability (lost on
  restart), no eviction, no locking (single-threaded `asyncio`, no `await` mid-mutation) -- with
  one caveat since milestone 7: `replace()`'s read-modify-write (via `CompactionService`) does
  have an `await` in the middle (the summarization call), so concurrent requests against the same
  `(agent, session_id)` are not safe against a lost update. Not a regression in `get`/`append`/
  `create`, which remain safe as before. The only implementation so far; a durable backend
  (Redis/SQLite) can be added later behind the same protocol without touching any caller.
