"""Protocol interfaces for LLM implementations."""

from typing import Any, Protocol

from agent.core.models.completion import Completion
from agent.core.models.message import Message


class ILLM(Protocol):
    """Interface for anything that can turn messages into a completion."""

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        """Send messages to an LLM and return its completion.

        `temperature`/`top_p`/`max_tokens` of `None` mean "use this implementation's
        configured default," not "omit this param from consideration entirely." `tools`,
        if given, is a list of OpenAI-format function schemas offered to the LLM; the LLM
        may respond with `tool_calls` on the returned `Completion.message` instead of (or
        without) `content` -- this call never executes them.

        Contractual requirement on implementations: a request declaring no `tools` must never
        carry `role="tool"` messages or `tool_calls`-bearing messages (Bedrock's Converse API
        rejects such a request outright), so an implementation folds any such content out of
        `messages` itself -- e.g. via `flatten_tool_exchanges_for_no_tools_request` in
        `core/models/message.py`. Callers may therefore always pass real, un-pre-processed
        history, whether or not this particular call declares tools.
        """
        ...

    def max_input_tokens(self) -> int:
        """Return this model's maximum input token count.

        Not async -- a local model-data lookup, no network call.

        Raises:
            LLMError: if the underlying provider has no known limit for this model.
        """
        ...
