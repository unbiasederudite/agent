# api

Inbound HTTP adapter. Thin translation layer between HTTP and `core/services/` — contains no reasoning of its own.

## Contents

- `schemas.py` — OpenAI-compatible request/response models for `/v1/chat/completions`
- `app.py` — `create_app(config_path)` builds the FastAPI app from an `AppConfig` JSON file; `add_chat_completions_route()` registers the route on a given app instance
- `__main__.py` — CLI entrypoint: parses `--config`/`--host`/`--port` and starts the server

## Running

    uv run python -m agent.api --config path/to/config.json

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.
