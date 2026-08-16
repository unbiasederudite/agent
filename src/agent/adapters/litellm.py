"""LiteLLM adapter implementation for ILLM protocol."""

import litellm

from agent.core.exceptions import LLMError, LLMRateLimitedError, LLMTimeoutError
from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage


class LiteLLMAdapter:
    """ILLM implementation backed by litellm, supporting any litellm provider."""

    def __init__(self, model: str) -> None:
        """Initialize adapter with a model identifier.

        Args:
            model: The model string (e.g., 'openai/gpt-4o') to use with litellm.
        """
        self._model = model

    async def complete(self, messages: list[Message]) -> Completion:
        """Send messages to litellm and map the result to a Completion.

        Raises:
            LLMRateLimitedError: if litellm reports the provider rate-limited the request.
            LLMTimeoutError: if litellm reports the request timed out.
            LLMError: if the underlying litellm call fails for any other reason.
        """
        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[m.model_dump() for m in messages],
            )
            choice = response.choices[0]
            return Completion(
                message=Message(role="assistant", content=choice.message.content),
                usage=Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                finish_reason=choice.finish_reason,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 429:
                raise LLMRateLimitedError(str(exc)) from exc
            if status_code == 408:  # litellm's own marker for a timeout, not real HTTP 408
                raise LLMTimeoutError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
