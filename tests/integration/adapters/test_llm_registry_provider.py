from unittest.mock import AsyncMock

import litellm
import pytest
from guardrails.stores.context import set_call_kwargs

from agent.adapters.llm_registry_provider import (
    LLM_REGISTRY_PROVIDER,
    LLMRegistryProvider,
    guardrail_call_scope,
    register_llm_registry_provider,
)
from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage
from agent.core.registries.llm import LLMRegistry
from agent.core.run_context import collect_extra_usage, run_context


def _completion(content: str = "the answer") -> Completion:
    return Completion(
        message=Message(role="assistant", content=content),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def test_completion_given_registered_model_calls_its_illm_and_returns_content():
    llm = AsyncMock()
    llm.complete.return_value = _completion("hello from the registry")
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    response = handler.completion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.2},
    )

    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "hello from the registry"
    assert response.choices[0].finish_reason == "stop"


def test_completion_given_registered_model_passes_messages_and_sampling_params():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    handler.completion(
        model="openai/gpt-4o",
        messages=[{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.5, "top_p": 0.9, "max_tokens": 100},
    )

    call_args, call_kwargs = llm.complete.call_args
    assert call_args[0] == [
        Message(role="system", content="be helpful"),
        Message(role="user", content="hi"),
    ]
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["max_tokens"] == 100


async def test_acompletion_given_registered_model_calls_its_illm_and_returns_content():
    llm = AsyncMock()
    llm.complete.return_value = _completion("async hello")
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    response = await handler.acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.2},
    )

    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "async hello"
    assert response.choices[0].finish_reason == "stop"


async def test_acompletion_given_unregistered_model_raises_llm_not_found_error():
    from agent.core.exceptions import LLMNotFoundError

    registry = LLMRegistry()
    handler = LLMRegistryProvider(registry)

    with pytest.raises(LLMNotFoundError):
        await handler.acompletion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )


def test_completion_given_unregistered_model_raises_llm_not_found_error():
    from agent.core.exceptions import LLMNotFoundError

    registry = LLMRegistry()
    handler = LLMRegistryProvider(registry)

    with pytest.raises(LLMNotFoundError):
        handler.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}])


def test_register_llm_registry_provider_given_no_existing_map_registers_it():
    litellm.custom_provider_map = None
    registry = LLMRegistry()

    register_llm_registry_provider(registry)

    assert litellm.custom_provider_map is not None
    providers = [entry["provider"] for entry in litellm.custom_provider_map]
    assert LLM_REGISTRY_PROVIDER in providers


def test_register_llm_registry_provider_given_existing_map_appends_not_replaces():
    other_handler = object()
    litellm.custom_provider_map = [{"provider": "something-else", "custom_handler": other_handler}]
    registry = LLMRegistry()

    register_llm_registry_provider(registry)

    providers = {entry["provider"] for entry in litellm.custom_provider_map}
    assert providers == {"something-else", LLM_REGISTRY_PROVIDER}


def test_register_llm_registry_provider_given_already_registered_keeps_single_entry():
    registry = LLMRegistry()
    register_llm_registry_provider(registry)

    register_llm_registry_provider(registry)

    providers = [entry["provider"] for entry in litellm.custom_provider_map]
    assert providers.count(LLM_REGISTRY_PROVIDER) == 1


def test_register_llm_registry_provider_given_new_registry_rebinds_to_it():
    llm_a = AsyncMock()
    llm_a.complete.return_value = _completion("from registry A")
    registry_a = LLMRegistry()
    registry_a.register("openai/gpt-4o", llm_a)
    register_llm_registry_provider(registry_a)

    llm_b = AsyncMock()
    llm_b.complete.return_value = _completion("from registry B")
    registry_b = LLMRegistry()
    registry_b.register("openai/gpt-4o", llm_b)
    register_llm_registry_provider(registry_b)

    response = litellm.completion(
        model=f"{LLM_REGISTRY_PROVIDER}/openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.choices[0].message.content == "from registry B"
    llm_a.complete.assert_not_called()


def test_completion_given_active_run_records_usage_into_it():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    with run_context("researcher", "sess-1"):
        handler.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert collect_extra_usage().total_tokens == 2


async def test_acompletion_given_active_run_records_usage_into_it():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    with run_context("researcher", "sess-1"):
        await handler.acompletion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )

        assert collect_extra_usage().total_tokens == 2


def test_get_llm_provider_given_registered_model_resolves_without_calling_completion():
    registry = LLMRegistry()
    register_llm_registry_provider(registry)

    model, provider, dynamic_key, api_base = litellm.get_llm_provider(
        f"{LLM_REGISTRY_PROVIDER}/openai/gpt-4o-mini"
    )

    assert model == "openai/gpt-4o-mini"
    assert provider == LLM_REGISTRY_PROVIDER
    assert dynamic_key is None
    assert api_base is None


def test_get_llm_provider_given_unrelated_model_still_resolves_normally():
    registry = LLMRegistry()
    register_llm_registry_provider(registry)

    model, provider, _dynamic_key, _api_base = litellm.get_llm_provider("gpt-4o-mini")

    assert model == "gpt-4o-mini"
    assert provider == "openai"


def test_completion_end_to_end_through_litellm_routes_to_registered_llm():
    litellm.custom_provider_map = None
    llm = AsyncMock()
    llm.complete.return_value = _completion("routed correctly")
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    register_llm_registry_provider(registry)

    response = litellm.completion(
        model=f"{LLM_REGISTRY_PROVIDER}/openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.choices[0].message.content == "routed correctly"


def test_completion_given_open_guardrail_scope_defers_usage_to_scope_exit():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    with run_context("researcher", "sess-1"):
        with guardrail_call_scope() as call_kwargs:
            set_call_kwargs(call_kwargs)
            handler.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}])
            # Simulates being inside Guard.parse()'s isolated context: not yet visible here.
            assert collect_extra_usage().total_tokens == 0

        assert collect_extra_usage().total_tokens == 2


async def test_acompletion_given_open_guardrail_scope_defers_usage_to_scope_exit():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    with run_context("researcher", "sess-1"):
        with guardrail_call_scope() as call_kwargs:
            set_call_kwargs(call_kwargs)
            await handler.acompletion(
                model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )
            assert collect_extra_usage().total_tokens == 0

        assert collect_extra_usage().total_tokens == 2


def test_guardrail_call_scope_given_no_active_run_drops_usage_without_error():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    with guardrail_call_scope() as call_kwargs:
        set_call_kwargs(call_kwargs)
        handler.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}])

    with run_context("researcher", "sess-1"):
        assert collect_extra_usage().total_tokens == 0


def test_guardrail_call_scope_given_two_concurrent_scopes_keeps_usage_separate():
    llm = AsyncMock()
    llm.complete.return_value = _completion()
    registry = LLMRegistry()
    registry.register("openai/gpt-4o", llm)
    handler = LLMRegistryProvider(registry)

    with run_context("researcher", "sess-1"):
        with guardrail_call_scope(), guardrail_call_scope() as inner_kwargs:
            set_call_kwargs(inner_kwargs)
            handler.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}])
            assert collect_extra_usage().total_tokens == 0

        assert collect_extra_usage().total_tokens == 2
