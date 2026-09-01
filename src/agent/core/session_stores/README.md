# session_stores

`ISessionStore` (`core/protocols/isession_store.py`) implementations: per-conversation message
history storage, keyed by `(agent, session_id)`.

## Contents

- `in_memory.py` — `InMemorySessionStore`, backed by a process-local dict. No durability (lost on
  restart). `get`/`append`/`create` need no locking (single-threaded `asyncio`, no
  `await` mid-mutation), but a read-modify-write spanning more than one call (e.g.
  `CompactionService.compact()`'s get-summarize-replace, which has an `await` in the middle for
  the summarization call) is not safe against a lost update on its own -- `lock()` closes that
  gap: a lazily-created `dict[tuple[str, str], asyncio.Lock]`, one `asyncio.Lock` per
  `(agent, session_id)`, created on first use. Lock contention (a caller finding the lock already
  held) logs at DEBUG. `busy()` is a separate, non-blocking `set[tuple[str, str]]`-backed
  test-and-set: raises `SessionBusyError` immediately (never waits) if the pair is already
  marked, for rejecting a whole second concurrent operation outright rather than serializing it.
  `delete()` removes a session's stored history and its lock entirely. Now bounded: `_sessions`
  and `_locks` are each capped at the constructor's `max_sessions` (`None` means unbounded) via
  LRU eviction keyed on last-touched order, skipping any session currently locked or busy so an
  actively in-flight session is never evicted out from under its own operation -- see
  `_evict_if_over_capacity`. A durable backend (Redis/SQLite) can be added later behind the same
  protocol without touching any caller.
