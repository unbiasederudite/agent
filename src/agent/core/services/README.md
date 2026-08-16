# services

Use-case orchestration, called by inbound adapters.

## Contents

- `completion.py` — `CompletionService`, resolves an agent and/or LLM by name and runs a single chat completion, prepending the agent's system prompt and resolving sampling defaults when an agent is selected
