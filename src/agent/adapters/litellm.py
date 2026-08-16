"""LiteLLM adapter implementation for ILLM protocol."""

import litellm

from agent.core.exceptions import LLMError, LLMRateLimitedError, LLMTimeoutError
from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`."""
    return a if a is not None else b


class LiteLLMAdapter:
    """ILLM implementation backed by litellm, supporting any litellm provider."""

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize adapter with a model identifier and its configured sampling defaults.

        Args:
            model: The model string (e.g., 'openai/gpt-4o') to use with litellm.
            temperature: Default sampling temperature, used when a call doesn't override it.
            top_p: Default nucleus sampling value, used when a call doesn't override it.
            max_tokens: Default max output tokens, used when a call doesn't override it.
        """
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Send messages to litellm and map the result to a Completion.

        `temperature`/`top_p`/`max_tokens` of `None` fall back to this adapter's
        constructed default; if that is also `None`, the param is omitted from the
        litellm call and the provider's own default applies.

        Raises:
            LLMRateLimitedError: if litellm reports the provider rate-limited the request.
            LLMTimeoutError: if litellm reports the request timed out.
            LLMError: if the underlying litellm call fails for any other reason.
        """
        resolved_temperature = _first_not_none(temperature, self._temperature)
        resolved_top_p = _first_not_none(top_p, self._top_p)
        resolved_max_tokens = _first_not_none(max_tokens, self._max_tokens)

        params: dict[str, float | int] = {
            key: value
            for key, value in (
                ("temperature", resolved_temperature),
                ("top_p", resolved_top_p),
                ("max_tokens", resolved_max_tokens),
            )
            if value is not None
        }

        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[m.model_dump() for m in messages],
                **params,
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
