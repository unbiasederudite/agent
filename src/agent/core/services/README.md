# services

Use-case orchestration, called by inbound adapters.

## Contents

- `completion.py` — `CompletionService`, resolves an agent and/or LLM by name and runs a single chat completion, prepending the agent's system prompt, resolving sampling defaults, and resolving the tri-state `tools` selection (request overrides agent, empty list suppresses) into OpenAI-format function schemas
