# protocols

Interfaces for anything with interchangeable implementations.

## Contents

- `illm.py` — `ILLM`, implemented by any outbound adapter that can turn messages into a completion (currently: `LiteLLMAdapter`). `complete()` takes optional `temperature`/`top_p`/`max_tokens`; `None` means "use the implementation's own configured default."
