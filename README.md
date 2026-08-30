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
prefix. An optional top-level `compaction` block keeps a session's stored history under a
token budget by summarizing its older portion once it grows too large -- see
`config.example.json` and `src/agent/core/models/README.md` for its fields; omit it entirely
to disable the feature.

Then start the API:

```bash
uv run python -m agent.api --config config.json
```

`--host` (default `127.0.0.1`) and `--port` (default `8000`) are optional.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/agents/{agent_name}` | Run `agent_name` against `message`. Body: `message` (required, a plain string), `model`/`strategy`/`temperature`/`top_p`/`max_tokens` (optional overrides), `tools` (optional, tri-state: omitted uses the agent's configured tools, `[]` suppresses them, a list overrides them), `session_id` (optional -- omit to start a new conversation, pass a prior response's `session_id` back to continue it). If the LLM requests a tool call, it is executed and fed back automatically (up to the agent's `max_tool_iterations`) before a response is returned. Returns `model`, `message`, `usage`, `finish_reason`, `session_id`. |
| `GET` | `/v1/agents` | List registered agents: `name`, `model`, `strategy`, `tools`. |
| `GET` | `/v1/tools` | List registered tools: `name`, `description`, `parameters` (JSON schema). |
| `GET` | `/v1/models` | List registered model id strings. |
| `GET` | `/v1/strategies` | List registered reasoning strategy names (valid values for `strategy`). |

## Errors

Every error response body is `{"detail": {"message": "...", "code": "..."}}` (validation
errors from FastAPI itself, and the 500 catch-all, omit `code`). Notable `code` values on
`POST /v1/agents/{agent_name}`:

| Status | `code` | Meaning |
|---|---|---|
| 404 | `agent_not_found` / `model_not_found` / `strategy_not_found` / `session_not_found` / `tool_not_found` | The named resource isn't registered (or, for `session_not_found`, doesn't belong to this agent). |
| 400 | `input_too_large` | `message` exceeds the agent's configured `max_input_chars`. |
| 429 | *(none)* | The provider rate-limited the request. |
| 504 | *(none)* | The provider request timed out. |
| 502 | `context_window_exceeded` | The request overflowed the model's context window and compaction was never available to try (not configured, or a brand-new session with no history to compact). |
| 502 | `compaction_exhausted` | The request overflowed and compaction *was* tried but couldn't help -- see `CompactionExhaustedError`'s docstring for what that does and doesn't mean for the session's stored history afterward. |
| 502 | *(none)* | Any other underlying LLM call failure. |
| 500 | *(none)* | An otherwise-unhandled `AgentError`, or any unexpected exception. |
