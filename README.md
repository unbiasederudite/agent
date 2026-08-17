# agent-core

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

`config.json` declares which litellm-format `provider/model` strings are allowed, and the
agents (see Endpoints below) available to route requests through;
`.env` holds the API key(s) litellm reads automatically based on each model's provider
prefix.

Then start the API:

```bash
uv run python -m agent.api --config config.json
```

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/agents/{agent_name}` | Run `agent_name` against `messages`. Body: `messages` (required), `model`/`temperature`/`top_p`/`max_tokens` (optional overrides), `tools` (optional, tri-state: omitted uses the agent's configured tools, `[]` suppresses them, a list overrides them). Returns `model`, `message`, `usage`, `finish_reason`. |
| `GET` | `/v1/agents` | List registered agents: `name`, `default_llm`, `tools`. |
| `GET` | `/v1/tools` | List registered tools: `name`, `description`, `parameters` (JSON schema). |
| `GET` | `/v1/llms` | List registered model id strings. |
