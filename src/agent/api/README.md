# api

Inbound HTTP adapter. Thin translation layer between HTTP and `core/services/` — contains no reasoning of its own.

## Contents

- `schemas.py` — OpenAI-compatible request/response models for `/v1/chat/completions`; the request accepts an optional `agent` and optional `model` (at least one required), optional `temperature`/`top_p`/`max_completion_tokens` (OpenAI's current field name; the deprecated `max_tokens` is not accepted, silently ignored as an unrecognized field), and an optional tri-state `tools` (registered tool **names**, not OpenAI-format function-definition objects — a deliberate divergence, since the server only ever offers tools it has itself registered; omitted uses the agent's tools, `[]` suppresses them, a list overrides them). OpenAI's `tool_choice`/`parallel_tool_calls` request fields are not modeled and are silently ignored if sent. A request message's `tool_calls` is rejected (400) — this milestone has no `role: "tool"`/`tool_call_id` support to complete a replayed round trip. `ChatToolCall`/`ChatToolCallFunction` model an OpenAI-compatible tool call in the response. `ErrorDetail`/`ErrorResponse` model the OpenAI-compatible `{"error": {...}}` error envelope.
- `app.py` — `create_app(config_path)` builds the FastAPI app from an `AppConfig` JSON file; `add_chat_completions_route()` registers the route on a given app instance, plus three exception handlers (`RequestValidationError`, `StarletteHTTPException`, and a catch-all) that translate every error response (validation failures, `AgentError` subclasses, framework-level errors) into OpenAI's `{"error": {message, type, param, code}}` shape
- `__main__.py` — CLI entrypoint: parses `--config`/`--host`/`--port` and starts the server

## Running

    uv run python -m agent.api --config path/to/config.json

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Errors

Every error response — validation failures (400), `AgentError` subclasses (404/429/502/504/500),
and framework-level errors (unmatched routes, wrong HTTP method) — uses OpenAI's error envelope:
`{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`.
