# models

Pydantic data models: domain models and startup config.

## Contents

- `message.py` — `Message`, `ToolCall`/`ToolCallFunction`, and `flatten_tool_exchanges_for_no_tools_request`, the OpenAI-compatible chat message wire format and its tool-call helpers
- `usage.py` — `Usage`, `ZERO_USAGE`, and `sum_usage()`, token/cost accounting for one completion
- `completion.py` — `Completion`, the result of one `ILLM.complete()` call
- `turn.py` — `Turn`, the aggregate result of one reasoning-loop run
- `run.py` — `Run`, the domain record of one completion execution
- `config.py` — the startup config models loaded once from JSON: `SamplingDefaults`, `LLMConfig`, `AgentConfig`, `ToolConfig`, `StrategyConfig`, `CompactionConfig`, `LoggingConfig`, and root `AppConfig`
