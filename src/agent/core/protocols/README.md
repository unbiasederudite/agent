# protocols

Interfaces for anything with interchangeable implementations.

## Contents

- `illm.py` — `ILLM`, the interface for an outbound adapter that turns messages into a completion
- `itool.py` — `ITool`, the interface for a tool an agent can expose to an LLM
- `istrategy.py` — `IStrategy`, the interface for a reasoning-loop algorithm
- `iguardrail.py` — `IGuardrail` and `run_guardrails()`, the interface for a content check run at a guardrail checkpoint and the shared pipeline that runs a list of them against content, applying each one's configured action
- `isession_store.py` — `ISessionStore`, the interface for per-conversation history storage
