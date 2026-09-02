"""LiteLLM adapter implementation for ILLM protocol."""

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
    """Return `a`, or `b` if `a` is `None`."""
    return a if a is not None else b


def _is_retriable(status_code: int | None) -> bool:
    """No HTTP response at all (a raw connection error) or a 5xx is worth retrying.

    Any other 4xx (bad request, auth, permission, not found, unprocessable) is a permanent,
    client-side mistake that would fail identically on a second attempt.
    """
    return status_code is None or status_code >= 500


def _classify(exc: Exception, status_code: int | None) -> LLMError:
    """Map a caught provider exception to its typed equivalent, by status code."""
    if status_code == 429:
        return LLMRateLimitedError(str(exc))
    if status_code == 408:  # litellm's own marker for a timeout, not real HTTP 408
        return LLMTimeoutError(str(exc))
    return LLMError(str(exc))


class LiteLLMAdapter:
    """ILLM implementation backed by litellm, supporting any litellm provider."""

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
            model: The model string (e.g., 'openai/gpt-4o') to use with litellm.
            temperature: Default sampling temperature, used when a call doesn't override it.
            top_p: Default nucleus sampling value, used when a call doesn't override it.
            max_tokens: Default max output tokens, used when a call doesn't override it.
            context_window: Overrides litellm's own context-window lookup for this model
                in `max_input_tokens()`; used when litellm's static data doesn't recognize
                this model. `None` leaves the litellm lookup in place.
            num_retries: Retries for retriable failures before giving up. This adapter runs
                its own retry loop -- litellm's own `num_retries` parameter is never used,
                since it would retry permanent errors (bad auth, bad request) as eagerly as
                transient ones.
            timeout: Per-attempt timeout in seconds, forwarded to litellm. `None` leaves
                litellm's own default in place.
            retry_base_delay: Delay before the first retry, in seconds.
            retry_max_delay: Cap on delay between retries, in seconds.
            retry_multiplier: Backoff multiplier applied to the delay after each retry.
            max_concurrent_requests: Cap on concurrent in-flight calls to this model.
                `None` means unlimited.
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
        """Build an adapter from a startup `LLMConfig`, one field-for-field mapping.

        The only place `LLMConfig`'s fields are unpacked into `__init__`'s matching
        parameters -- callers building an adapter from config (`core/factories/app.py`)
        use this instead of repeating that unpacking themselves.
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

        `temperature`/`top_p`/`max_tokens` of `None` fall back to this adapter's
        constructed default; if that is also `None`, the param is omitted from the
        litellm call and the provider's own default applies. `tools`, if non-empty, is
        forwarded to litellm as-is; any `tool_calls` litellm returns are mapped onto the
        returned message and never invoked here.

        Whenever `tools` is empty (`None` or `[]`), `messages` is first folded through
        `flatten_tool_exchanges_for_no_tools_request`: some providers -- Bedrock's Converse
        API, confirmed -- reject a request carrying `role="tool"` messages or `tool_calls`
        when it declares no `tools`, even when those merely replay an earlier exchange. This
        is unconditional and independent of why the caller has no tools this time, so callers
        may always pass real, un-pre-processed history. It is a no-op for a list that holds no
        tool-shaped content.

        Retries retriable failures (rate limits, timeouts, 5xx, connection errors) up to
        `num_retries` times with exponential backoff before raising; a permanent failure
        (4xx other than 429/408) raises immediately, never retried.

        Raises:
            LLMOverloadedError: if `max_concurrent_requests` is set and already reached --
                checked before any provider call, never retried.
            LLMRateLimitedError: if litellm reports the provider rate-limited the request.
            LLMTimeoutError: if litellm reports the request timed out.
            LLMError: if the underlying litellm call fails for any other reason, or
                returns a message with no `tool_calls` and empty/`None` `content` --
                an empty-string reply would otherwise be stored verbatim and later
                rejected by Anthropic/Bedrock, which reject empty text blocks. An
                empty-string `content` is normalized to `None` even when `tool_calls`
                is present (so it isn't raised on) -- otherwise the empty string
                survives `model_dump(exclude_none=True)` and can trip the same
                empty-text-block rejection on a later call that replays this message.
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
        """The retry loop from Task 4 -- unchanged, just extracted so `complete()` can wrap it.

        This lets `complete()` add the concurrency check above without duplicating this body.
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

            # Parsing/constructing the result from a *successful* network call -- kept
            # outside the except block above so a shape problem here (empty `choices`,
            # a missing field) is classified as LLMError directly, once, rather than
            # falling into the retry-eligible except above: it has no .status_code, so
            # _is_retriable(None) would otherwise treat it as a transient failure and
            # silently retry a deterministically-malformed response num_retries times
            # before finally reporting it -- same wrong outcome, wasted retry budget.
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
                    # Rounded to clean up binary-float noise (e.g. 6.5999999999999995e-06
                    # instead of 6.6e-06) -- 10dp is far below any realistic per-call cost
                    # while still well above where that noise appears.
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
        raise AssertionError("unreachable -- the loop above always returns or raises")

    def max_input_tokens(self) -> int:
        """Return this model's maximum input token count.

        Not async -- a local model-data lookup, no network call. If this adapter was
        constructed with `context_window`, that value is returned directly and
        unconditionally, taking precedence over the litellm lookup even for a model
        litellm would otherwise recognize correctly.

        Raises:
            LLMError: if litellm has no known limit for this model.
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
