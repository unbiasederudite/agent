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

`config.json` declares which litellm-format `provider/model` strings are allowed;
`.env` holds the API key(s) litellm reads automatically based on each model's provider
prefix.

Then start the API:

```bash
uv run python -m agent.api --config config.json
```

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

`POST /v1/chat/completions` accepts an OpenAI-compatible request body. Provide `model` (any
litellm-format provider/model string declared in your config), `agent` (any agent name declared
in your config), or both — at least one is required. Selecting an agent prepends its configured
system prompt and, unless overridden, uses its `default_llm` and sampling defaults. `model`,
`temperature`, `top_p`, and `max_tokens` in the request always override the agent's/config's
defaults when given. If the request's `messages` includes its own `role: "system"` entry, it is
not merged with or overridden by the agent's system prompt — the agent's system prompt is always
prepended unconditionally, and the client's message passes through unchanged underneath it as
ordinary conversation content.
