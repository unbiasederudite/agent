# agent

A lightweight, extensible Python core for building AI agents.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv sync
uv run pre-commit install
```

## Running

Copy the example config and env files, and fill in your provider API key(s):

```bash
cp config.example.json config.json
cp .env.example .env
```

`config.json` is the process's full startup configuration — which models, tools, strategies,
and agents are available, plus optional settings like compaction and logging. See
`config.example.json` for a filled-in example and `src/agent/core/models/README.md` for
every field. `.env` holds the provider API key(s) litellm reads automatically.

Then start the API:

```bash
uv run python -m agent.api --config config.json
```

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Endpoints

With the server running, browse the full interactive API reference at
`http://127.0.0.1:8000/docs` (Swagger UI) or `/redoc` — every route, request/response schema,
and field description, generated live from the API.
