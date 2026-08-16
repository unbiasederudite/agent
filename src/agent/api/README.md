# api

Inbound HTTP adapter. Thin translation layer between HTTP and `core/services/` — contains no reasoning of its own.

## Contents

- `schemas.py` — OpenAI-compatible request/response models for `/v1/chat/completions`; the request accepts an optional `agent` and optional `model` (at least one required), plus optional `temperature`/`top_p`/`max_tokens`. `ErrorDetail`/`ErrorResponse` model the OpenAI-compatible `{"error": {...}}` error envelope.
- `app.py` — `create_app(config_path)` builds the FastAPI app from an `AppConfig` JSON file; `add_chat_completions_route()` registers the route on a given app instance, plus two exception handlers that translate every error response (validation failures, `AgentError` subclasses) into OpenAI's `{"error": {message, type, param, code}}` shape
- `__main__.py` — CLI entrypoint: parses `--config`/`--host`/`--port` and starts the server

## Running

    uv run python -m agent.api --config path/to/config.json

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Errors

Every error response — validation failures (400), `AgentError` subclasses (404/429/502/504/500),
and framework-level errors (unmatched routes, wrong HTTP method) — uses OpenAI's error envelope:
`{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`.
