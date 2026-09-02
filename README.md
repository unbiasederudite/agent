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
| `POST` | `/v1/agents/{agent_name}` | Run `agent_name` against `message`. Body: `message` (required, a plain string), `model`/`strategy`/`temperature`/`top_p`/`max_tokens` (optional overrides), `tools` (optional, tri-state: omitted uses the agent's configured tools, `[]` suppresses them, a list overrides them), `session_id` (optional -- omit to start a new conversation, pass a prior response's `session_id` back to continue it). If the LLM requests a tool call, it is executed and fed back automatically (up to the agent's `max_tool_iterations`) before a response is returned. Returns `model`, `message`, `usage`, `finish_reason`, `session_id`. Rejects a concurrent second request against the same `session_id` outright (see Errors' `session_busy` row) rather than letting it run against stale history. |
| `GET` | `/v1/agents/{agent_name}/sessions/{session_id}` | Return the full stored history for this session: `session_id`, `messages` (every stored message, in order, unfiltered -- every role, including tool calls and tool results exactly as stored). |
| `GET` | `/v1/agents/{agent_name}/sessions/{session_id}/usage` | Return token/cost usage for this session: `session_id`, `cumulative` (`Usage` summed across every run against this session), `context_tokens` (footprint of the full stored history as of the last run, `0` if not currently known). |
| `DELETE` | `/v1/agents/{agent_name}/sessions/{session_id}` | Permanently remove this session and its stored history. `204` on success, no body. |
| `GET` | `/v1/agents` | List registered agents: `name`, `model`, `strategy`, `tools`. |
| `GET` | `/v1/agents/{agent_name}/usage` | Return cumulative token/cost usage for this agent across all its sessions: `agent`, `cumulative` (`Usage`, all-zero if never run). |
| `GET` | `/v1/tools` | List registered tools: `name`, `description`, `parameters` (JSON schema). |
| `GET` | `/v1/models` | List registered model id strings. |
| `GET` | `/v1/strategies` | List registered reasoning strategy names (valid values for `strategy`). |
| `GET` | `/health` | Liveness check only -- no dependency/provider checks, not logged. |

## Errors

Every error response body is `{"detail": {"message": "..."}}`, `code` added when the
status alone doesn't disambiguate the cause (validation errors from FastAPI itself, the
500 catch-all, and Starlette's own framework-raised errors -- unmatched route, wrong
method -- omit `code`; the latter two are normalized into this same `{"message": ...}`
shape by `handle_http_exception` even though Starlette's own default `detail` for them is
a bare string). Notable `code` values on `POST /v1/agents/{agent_name}`:

| Status | `code` | Meaning |
|---|---|---|
| 404 | `agent_not_found` / `model_not_found` / `strategy_not_found` / `session_not_found` / `tool_not_found` | The named resource isn't registered (or, for `session_not_found`, doesn't belong to this agent). |
| 413 | `input_too_large` | `message` exceeds the agent's configured `max_input_chars` (RFC 9110 section 15.5.14 -- content larger than the server is willing to process, not a generic 400). Also logged once at INFO in `agent_run.py` when raised -- `handle_http_exception` does not log it a second time. |
| 429 | *(none)* | The provider rate-limited the request. The message is a fixed, human-authored string, not the raw upstream error. Response carries a `Retry-After` header (a fixed conservative value -- litellm doesn't surface a provider-supplied one). |
| 504 | *(none)* | `LLMTimeoutError` -- the provider request itself timed out. The message is a fixed, human-authored string, not the raw upstream error. |
| 504 | *(none)* | `RequestTimeoutError` -- the whole request exceeded the agent's configured `max_request_seconds` (adapter retries, tool-calling rounds, and any reactive compaction-and-retry, combined). Shares the 504 status with the `LLMTimeoutError` row above but is a distinct condition; neither carries a `code`, so distinguish them by message text. |
| 503 | *(none)* | `LLMOverloadedError` -- the configured `max_concurrent_requests` for this model was already reached; the caller should retry after a short delay. Response carries a `Retry-After` header, same as the 429 row above. |
| 409 | `session_busy` | Another operation is already using this session. |
| 403 | `tool_not_allowed` / `model_not_allowed` / `strategy_not_allowed` | This agent does not permit the requested tool/model/strategy. |
| 413 | `context_window_exceeded` | The request overflowed the model's context window and compaction was never available to try (not configured, or a brand-new session with no history to compact). 413, not 502 -- the provider's rejection is a valid, well-formed "too large" response, not an invalid one (RFC 9110 section 15.6.3 reserves 502 for that), so it's the same 413 Content Too Large condition as `input_too_large` above. |
| 413 | `compaction_exhausted` | The request overflowed and compaction *was* tried but couldn't help -- see `CompactionExhaustedError`'s docstring for what that does and doesn't mean for the session's stored history afterward. Same 413 reasoning as `context_window_exceeded` above. |
| 502 | *(none)* | Any other underlying LLM call failure. The message is a fixed, human-authored string, not the raw upstream error; the body also includes a `request_id` field for cross-referencing against server-side logs. |
| 500 | *(none)* | An otherwise-unhandled `AgentError`, or any unexpected exception. The body also includes a `request_id` field for cross-referencing against server-side logs. |
| 404 / 405 | *(none)* | Unmatched route / wrong HTTP method, raised by Starlette itself rather than this app. |
