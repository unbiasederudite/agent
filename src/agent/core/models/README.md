# models

Pydantic data models.

## Contents

- `message.py` — `Message`, the OpenAI-compatible wire format for a single chat message (`role` includes `"tool"`, paired with `tool_call_id` and `name`, for tool-execution results); `ToolCall`/`ToolCallFunction`, one tool invocation an LLM requested, in litellm/OpenAI's nested wire shape (matches litellm's response exactly, so mapping and outbound serialization are both plain, direct operations)
- `usage.py` — `Usage`, token counts for one completion
- `completion.py` — `Completion`, the result of one `ILLM.complete()` wire call
- `turn.py` — `Turn`, the aggregate result of one `IStrategy.run()` call: every message it generated (`messages`), summed `usage`, and the terminal `finish_reason` — distinct from `Completion`, which is just one wire call's result
- `run.py` — `Run`, the domain record of one completion execution, including the `session_id` it belongs to
- `config.py` — `SamplingDefaults` (shared temperature/top_p/max_tokens fields), `LLMConfig`, `AgentConfig` (includes `strategy` and `max_tool_iterations`), `ToolConfig`, `StrategyConfig`, `LoggingConfig`, `AppConfig` (includes `base_prompt`, merged into every agent's leading system message): the startup config models, loaded once from JSON
