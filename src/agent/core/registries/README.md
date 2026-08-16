# registries

Runtime name-to-instance maps, populated by `factories/` at startup and queried by `services/` at request time.

## Contents

- `llm.py` — `LLMRegistry`, maps a litellm-format model string to its `ILLM` instance
