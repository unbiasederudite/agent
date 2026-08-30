# api

Inbound HTTP adapter. Thin translation layer between HTTP and `core/services/` — contains no reasoning of its own.

## Contents

- `schemas.py` — backend-native request/response models. `AgentRunRequest`/`AgentRunResponse` model `POST /v1/agents/{agent_name}` (`message` as a plain string, optional `model`/`strategy`/`temperature`/`top_p`/`max_tokens` overrides, tri-state `tools`: omitted uses the agent's tools, `[]` suppresses them, a list overrides them; optional `session_id` continues an existing conversation, omitted starts a new one). Tool calls requested by the LLM are executed server-side and fed back automatically (up to `AgentConfig.max_tool_iterations`). The response `message`/`usage` reuse `core.models.message.Message` and `core.models.usage.Usage` directly rather than duplicating identical DTOs — `models` is cross-cutting per `ARCHITECTURE.md`, so there's nothing to translate for those. `AgentRunResponse` stays its own model rather than reusing `core.models.run.Run` directly, keeping the API-facing DTO independent of the internal domain model. `AgentSummary`/`ToolSummary` model the `GET /v1/agents`/`GET /v1/tools` listing entries — deliberately reduced projections (`AgentSummary` omits `AgentConfig.system_prompt` and sampling defaults) rather than reusing the core models, since exposing those isn't safe.
- `app.py` — `create_app(config_path)` builds the FastAPI app from an `AppConfig` JSON file, constructing a `CompactionService` when the config declares one. `add_agent_run_route()` registers `POST /v1/agents/{agent_name}`, mapping `InputTooLargeError` to 400 and `LLMContextWindowExceededError` to 502 alongside the existing error mappings; `CompactionExhaustedError` (a subclass of `LLMContextWindowExceededError`, distinguishing "compaction was tried and could not help" from a plain overflow where compaction was never available) is checked *before* the generic `LLMContextWindowExceededError` handler -- `except` clauses run in order, and the more specific subclass must come first or its handler is unreachable. `add_registry_routes()` registers `GET /v1/agents`, `GET /v1/tools`, `GET /v1/models`, `GET /v1/strategies`; `add_exception_handlers()` registers a 400 handler for request-validation failures and a 500 catch-all — every other status (400/404/429/502/504/500 from `AgentError` subclasses, and framework 404/405s) is FastAPI's own `HTTPException`/Starlette handling. See the root `README.md`'s Errors section for the full status/`code` table.
- `__main__.py` — CLI entrypoint: parses `--config`/`--host`/`--port` and starts the server

## Running

    uv run python -m agent.api --config path/to/config.json

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Endpoints

See the root `README.md`'s Endpoints section for the full request/response contract.
