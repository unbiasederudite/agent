# protocols

Interfaces for anything with interchangeable implementations.

## Contents

- `illm.py` — `ILLM`, the interface for an outbound adapter that turns messages into a completion
- `itool.py` — `ITool`, the interface for a tool an agent can expose to an LLM
- `istrategy.py` — `IStrategy`, the interface for a reasoning-loop algorithm
- `isession_store.py` — `ISessionStore`, the interface for per-conversation history storage
