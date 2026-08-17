# models

Pydantic data models.

## Contents

- `message.py` — `Message`, the OpenAI-compatible wire format for a single chat message; `ToolCall`/`ToolCallFunction`, one tool invocation an LLM requested, in litellm/OpenAI's nested wire shape (matches litellm's response exactly, so mapping and outbound serialization are both plain, direct operations)
- `usage.py` — `Usage`, token counts for one completion
- `completion.py` — `Completion`, the result of an `ILLM.complete()` call
- `run.py` — `Run`, the domain record of one completion execution
- `config.py` — `SamplingDefaults` (shared temperature/top_p/max_tokens fields), `LLMConfig`, `AgentConfig`, `ToolConfig`, `LoggingConfig`, `AppConfig`: the startup config models, loaded once from JSON
