# registries

Runtime name-to-instance maps, populated by `factories/` at startup and queried by `services/` at request time.

## Contents

- `base.py` — `_Registry`, generic name-to-instance map shared by `LLMRegistry`/`AgentRegistry`, parameterized by item type and the exception `get()` raises on a miss
- `llm.py` — `LLMRegistry`, maps a litellm-format model string to its `ILLM` instance
- `agent.py` — `AgentRegistry`, maps an agent name to its `AgentConfig`
