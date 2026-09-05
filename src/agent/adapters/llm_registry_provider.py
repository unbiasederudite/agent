"""litellm custom-provider handler that redirects calls through a registered LLM."""

import asyncio
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import litellm
from guardrails.stores.context import get_call_kwarg
from litellm import CustomLLM  # type: ignore[attr-defined]

from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.protocols.illm import ILLM
from agent.core.registries.llm import LLMRegistry
from agent.core.run_context import record_extra_usage

LLM_REGISTRY_PROVIDER = "llmregistry"

_original_get_llm_provider = litellm.get_llm_provider

_GUARDRAIL_CALL_ID_KWARG = "_agent_core_guardrail_call_id"
_pending_guardrail_usage: dict[str, list[Usage]] = {}
_pending_guardrail_usage_lock = threading.Lock()


@contextmanager
def guardrail_call_scope() -> Iterator[dict[str, str]]:
    """Open a scope that recovers a supporting completion's usage from behind an isolation boundary.

    Yields:
        dict[str, str]: keyword arguments for the caller to splice into the isolated call.
    """
    call_id = uuid.uuid4().hex
    with _pending_guardrail_usage_lock:
        _pending_guardrail_usage[call_id] = []
    try:
        yield {_GUARDRAIL_CALL_ID_KWARG: call_id}
    finally:
        with _pending_guardrail_usage_lock:
            usages = _pending_guardrail_usage.pop(call_id, [])
        for usage in usages:
            record_extra_usage(usage)


def _record_usage(usage: Usage) -> None:
    """Attribute one completion's usage to whichever scope is waiting for it.

    Args:
        usage: The completion's usage to record.
    """
    call_id = get_call_kwarg(_GUARDRAIL_CALL_ID_KWARG)
    if call_id is not None:
        with _pending_guardrail_usage_lock:
            bucket = _pending_guardrail_usage.get(call_id)
            if bucket is not None:
                bucket.append(usage)
        return
    record_extra_usage(usage)


def _to_messages(raw_messages: list[dict[str, Any]]) -> list[Message]:
    """Convert litellm's OpenAI-format message dicts into this codebase's `Message` model.

    Args:
        raw_messages: litellm's call messages, as plain role/content dicts.

    Returns:
        list[Message]: the same messages, as `Message` instances.
    """
    return [Message(role=raw["role"], content=raw.get("content")) for raw in raw_messages]


async def _complete(
    llm: ILLM, messages: list[Message], optional_params: dict[str, Any]
) -> Completion:
    """Run one completion through `llm`, forwarding the sampling params litellm passed.

    Args:
        llm: The resolved LLM to call.
        messages: The call's messages.
        optional_params: litellm's `optional_params` dict (temperature, top_p, max_tokens,
            and any other provider kwargs a caller passed, which this bridge doesn't
            forward beyond those three).

    Returns:
        Completion: the call's result.
    """
    return await llm.complete(
        messages,
        temperature=optional_params.get("temperature"),
        top_p=optional_params.get("top_p"),
        max_tokens=optional_params.get("max_tokens"),
    )


def _to_model_response(result: Completion, model: str) -> litellm.ModelResponse:
    """Convert a `Completion` into litellm's own response shape.

    Args:
        result: The completion to convert.
        model: The model name litellm's response should report.

    Returns:
        litellm.ModelResponse: the call's result, in litellm's own response shape.
    """
    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.message.content},
                "finish_reason": result.finish_reason,
            }
        ],
        model=model,
        usage={
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    )


class LLMRegistryProvider(CustomLLM):
    """Routes a litellm call under the "llmregistry/" provider prefix to a registered LLM."""

    def __init__(self, llm_registry: LLMRegistry) -> None:
        """Initialize with the registry calls are routed through.

        Args:
            llm_registry: Registry of available LLM implementations.
        """
        super().__init__()
        self._llm_registry = llm_registry

    def completion(self, *args: Any, **kwargs: Any) -> litellm.ModelResponse:
        """Run one completion call through the registered LLM matching `kwargs["model"]`.

        Args:
            *args: Unused; litellm invokes this method by keyword.
            **kwargs: litellm's call arguments — `model` (with the provider prefix
                already stripped by litellm), `messages`, and `optional_params`.

        Returns:
            litellm.ModelResponse: the call's result, in litellm's own response shape.
        """
        llm = self._llm_registry.get(kwargs["model"])
        messages = _to_messages(kwargs["messages"])
        optional_params = kwargs.get("optional_params") or {}
        result = asyncio.run(_complete(llm, messages, optional_params))
        _record_usage(result.usage)
        return _to_model_response(result, kwargs["model"])

    async def acompletion(self, *args: Any, **kwargs: Any) -> litellm.ModelResponse:
        """Run one completion call through the registered LLM matching `kwargs["model"]`.

        Args:
            *args: Unused; litellm invokes this method by keyword.
            **kwargs: litellm's call arguments — `model` (with the provider prefix
                already stripped by litellm), `messages`, and `optional_params`.

        Returns:
            litellm.ModelResponse: the call's result, in litellm's own response shape.
        """
        llm = self._llm_registry.get(kwargs["model"])
        messages = _to_messages(kwargs["messages"])
        optional_params = kwargs.get("optional_params") or {}
        result = await _complete(llm, messages, optional_params)
        _record_usage(result.usage)
        return _to_model_response(result, kwargs["model"])


def _get_llm_provider_with_custom_providers(
    model: str,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    litellm_params: Any = None,
) -> tuple[str, str, str | None, str | None]:
    """Resolve a model string's provider, recognizing the "llmregistry/" prefix too.

    Args:
        model: Model string to resolve, e.g. "llmregistry/openai/gpt-4o-mini".
        custom_llm_provider: Caller-forced provider, if any.
        api_base: Provider API base override, if any.
        api_key: Provider API key override, if any.
        litellm_params: litellm's generic per-call parameter bag, if any.

    Returns:
        tuple[str, str, str | None, str | None]: model (provider prefix stripped),
        provider, dynamic API key, API base.
    """
    prefix = f"{LLM_REGISTRY_PROVIDER}/"
    if custom_llm_provider is None and model.startswith(prefix):
        return model[len(prefix) :], LLM_REGISTRY_PROVIDER, None, None
    return cast(
        "tuple[str, str, str | None, str | None]",
        _original_get_llm_provider(model, custom_llm_provider, api_base, api_key, litellm_params),
    )


def register_llm_registry_provider(llm_registry: LLMRegistry) -> None:
    """Register the "llmregistry/" custom litellm provider, rebinding it to `llm_registry`.

    Args:
        llm_registry: Registry of available LLM implementations.
    """
    other_providers = [
        entry
        for entry in (litellm.custom_provider_map or [])
        if entry["provider"] != LLM_REGISTRY_PROVIDER
    ]
    litellm.custom_provider_map = [
        *other_providers,
        {"provider": LLM_REGISTRY_PROVIDER, "custom_handler": LLMRegistryProvider(llm_registry)},
    ]
    litellm.get_llm_provider = _get_llm_provider_with_custom_providers
