# models

Pydantic data models.

## Contents

- `message.py` — `Message`, the OpenAI-compatible wire format for a single chat message
- `usage.py` — `Usage`, token counts for one completion
- `completion.py` — `Completion`, the result of an `ILLM.complete()` call
- `run.py` — `Run`, the domain record of one completion execution
- `config.py` — `LLMConfig`, `LoggingConfig`, `AppConfig`: the startup config models, loaded once from JSON
