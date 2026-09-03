"""LLM adapter backed by litellm, giving access to any provider litellm supports."""

import asyncio
import logging
import random
from typing import Any

import litellm

from agent.core.exceptions import (
    LLMContextWindowExceededError,
    LLMError,
    LLMOverloadedError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from agent.core.models.completion import Completion
from agent.core.models.config import LLMConfig
from agent.core.models.message import (
    Message,
    ToolCall,
    ToolCallFunction,
    flatten_tool_exchanges_for_no_tools_request,
)
from agent.core.models.usage import Usage

logger = logging.getLogger(__name__)


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`.

    Args:
        a: Preferred value.
        b: Fallback value.

    Returns:
        T | None: `a` if not `None`, else `b`.
    """
    return a if a is not None else b


def _is_retriable(status_code: int | None) -> bool:
    """Return whether a failure (by HTTP status) is worth retrying: no response, or a 5xx.

    Args:
        status_code: The failure's HTTP status code, if any.

    Returns:
        bool: whether the failure is worth retrying.
    """
    return status_code is None or status_code >= 500


def _classify(exc: Exception, status_code: int | None) -> LLMError:
    """Map a caught provider exception to its typed equivalent, by status code.

    Args:
        exc: The caught exception.
        status_code: The failure's HTTP status code, if any.

    Returns:
        LLMError: the typed equivalent.
    """
    if status_code == 429:
        return LLMRateLimitedError(str(exc))
    if status_code == 408:  # litellm's own marker for a timeout, not real HTTP 408
        return LLMTimeoutError(str(exc))
    return LLMError(str(exc))


class LiteLLMAdapter:
    """Turns messages into a completion via litellm, supporting any litellm provider."""

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        num_retries: int = 2,
        timeout: float | None = None,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_multiplier: float = 2.0,
        max_concurrent_requests: int | None = None,
    ) -> None:
        """Initialize adapter with a model identifier and its configured sampling defaults.

        Args:
            model: The litellm model string (e.g., 'openai/gpt-4o').
            temperature: Default sampling temperature.
            top_p: Default nucleus sampling value.
            max_tokens: Default max output tokens.
            context_window: Override for the model's context-window size.
            num_retries: Retry count for retriable failures.
            timeout: Per-attempt timeout in seconds.
            retry_base_delay: Delay before the first retry, in seconds.
            retry_max_delay: Cap on delay between retries, in seconds.
            retry_multiplier: Backoff multiplier for retry delay.
            max_concurrent_requests: Cap on concurrent in-flight calls to this model.
        """
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._num_retries = num_retries
        self._timeout = timeout
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._retry_multiplier = retry_multiplier
        self._max_concurrent = max_concurrent_requests
        self._in_flight = 0

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LiteLLMAdapter":
        """Build an adapter from a model's configuration, mapping its fields onto `__init__`.

        Args:
            config: The model's startup configuration.

        Returns:
            LiteLLMAdapter: the built adapter.
        """
        return cls(
            config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            context_window=config.context_window,
            num_retries=config.num_retries,
            timeout=config.timeout,
            retry_base_delay=config.retry_base_delay,
            retry_max_delay=config.retry_max_delay,
            retry_multiplier=config.retry_multiplier,
            max_concurrent_requests=config.max_concurrent_requests,
        )

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        """Send messages to litellm and map the result to a Completion.

        Args:
            messages: Conversation history to send.
            temperature: Temperature override for this call.
            top_p: Top-p override for this call.
            max_tokens: Max-tokens override for this call.
            tools: OpenAI-format function schemas to offer the model.

        Returns:
            Completion: the mapped model response.

        Raises:
            LLMOverloadedError: if the concurrency cap is already reached.
            LLMRateLimitedError: if the provider rate-limited the request.
            LLMTimeoutError: if the request timed out.
            LLMError: if the call fails, or the response has neither content nor tool_calls.
        """
        if self._max_concurrent is not None and self._in_flight >= self._max_concurrent:
            logger.warning(
                "model %s at capacity (%d/%d), rejecting",
                self._model,
                self._in_flight,
                self._max_concurrent,
                extra={
                    "model": self._model,
                    "in_flight": self._in_flight,
                    "max_concurrent": self._max_concurrent,
                },
            )
            raise LLMOverloadedError(
                f"model '{self._model}' is at capacity ({self._max_concurrent} concurrent requests)"
            )
        self._in_flight += 1
        try:
            return await self._complete_with_retries(
                messages, temperature, top_p, max_tokens, tools
            )
        finally:
            self._in_flight -= 1

    async def _complete_with_retries(
        self,
        messages: list[Message],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> Completion:
        """Send messages to litellm with retries, and map the result to a Completion.

        Args:
            messages: Conversation history to send.
            temperature: Temperature override for this call.
            top_p: Top-p override for this call.
            max_tokens: Max-tokens override for this call.
            tools: OpenAI-format function schemas to offer the model.

        Returns:
            Completion: the mapped model response.

        Raises:
            LLMRateLimitedError: if the provider rate-limited the request.
            LLMTimeoutError: if the request timed out.
            LLMError: if the call fails, or the response has neither content nor tool_calls.
        """
        resolved_temperature = _first_not_none(temperature, self._temperature)
        resolved_top_p = _first_not_none(top_p, self._top_p)
        resolved_max_tokens = _first_not_none(max_tokens, self._max_tokens)

        params: dict[str, Any] = {
            key: value
            for key, value in (
                ("temperature", resolved_temperature),
                ("top_p", resolved_top_p),
                ("max_completion_tokens", resolved_max_tokens),
                ("timeout", self._timeout),
            )
            if value is not None
        }
        if tools:
            params["tools"] = tools
        else:
            messages = flatten_tool_exchanges_for_no_tools_request(messages)

        outbound_messages = [m.model_dump(exclude_none=True) for m in messages]

        for attempt in range(self._num_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=self._model,
                    messages=outbound_messages,
                    **params,
                )
            except Exception as exc:
                if isinstance(exc, litellm.ContextWindowExceededError):  # type: ignore[attr-defined]
                    raise LLMContextWindowExceededError(str(exc)) from exc
                status_code = getattr(exc, "status_code", None)
                classified = _classify(exc, status_code)
                retriable = status_code in (429, 408) or _is_retriable(status_code)
                if not retriable or attempt == self._num_retries:
                    logger.error(
                        "LLM call failed permanently: %s (status=%s)",
                        type(exc).__name__,
                        status_code,
                        exc_info=True,
                        extra={
                            "exception_type": type(exc).__name__,
                            "status_code": status_code,
                            "attempt": attempt + 1,
                        },
                    )
                    raise classified from exc
                delay = min(
                    self._retry_base_delay * (self._retry_multiplier**attempt),
                    self._retry_max_delay,
                ) * random.uniform(0.5, 1.0)
                logger.warning(
                    "LLM call failed: %s (status=%s), attempt %d/%d, retrying in %.1fs: %s",
                    type(exc).__name__,
                    status_code,
                    attempt + 1,
                    self._num_retries + 1,
                    delay,
                    exc,
                    extra={
                        "exception_type": type(exc).__name__,
                        "status_code": status_code,
                        "attempt": attempt + 1,
                    },
                )
                await asyncio.sleep(delay)
                continue

            # Parsing/constructing the result from a *successful* call is kept outside the
            # except block above: a response-shape problem here has no .status_code, so
            # treating it as retriable would silently retry a deterministically-malformed
            # response num_retries times before reporting it, instead of failing once.
            try:
                choice = response.choices[0]
                raw_tool_calls = getattr(choice.message, "tool_calls", None)
                tool_calls = (
                    [
                        ToolCall(
                            id=tc.id,
                            function=ToolCallFunction(
                                name=tc.function.name, arguments=tc.function.arguments
                            ),
                        )
                        for tc in raw_tool_calls
                    ]
                    if raw_tool_calls
                    else None
                )
                content = choice.message.content or None
                if content is None and tool_calls is None:
                    raise LLMError("litellm returned a message with neither content nor tool_calls")
                try:
                    cost_usd = round(litellm.completion_cost(completion_response=response), 10)
                except Exception:
                    cost_usd = None
                return Completion(
                    message=Message(role="assistant", content=content, tool_calls=tool_calls),
                    usage=Usage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        cost_usd=cost_usd,
                    ),
                    finish_reason=choice.finish_reason,
                )
            except LLMError as malformed:
                logger.error(
                    "provider returned a malformed response: %s",
                    malformed,
                    exc_info=True,
                    extra={"exception_type": type(malformed).__name__},
                )
                raise
            except (IndexError, AttributeError, KeyError, TypeError) as exc:
                shape_error = LLMError(f"litellm returned an unexpected response shape: {exc}")
                logger.error(
                    "provider returned a malformed response: %s",
                    shape_error,
                    exc_info=True,
                    extra={"exception_type": type(shape_error).__name__},
                )
                raise shape_error from exc
        raise AssertionError("unreachable — the loop above always returns or raises")

    def max_input_tokens(self) -> int:
        """Return this model's maximum input token count.

        Returns:
            int: the model's maximum input token count.

        Raises:
            LLMError: if no known limit is found for this model.
        """
        if self._context_window is not None:
            return self._context_window
        try:
            info = litellm.get_model_info(self._model)
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        max_input = info.get("max_input_tokens")
        if max_input is None:
            raise LLMError(f"litellm has no max_input_tokens for model '{self._model}'")
        return max_input
