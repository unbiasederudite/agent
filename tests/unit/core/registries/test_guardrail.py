"""Tests for GuardrailRegistry."""

import pytest

from agent.core.exceptions import GuardrailNotFoundError
from agent.core.models.guardrail import GuardrailFinding
from agent.core.registries.guardrail import GuardrailRegistry


class _FakeGuardrail:
    name = "no-secrets"
    action = "block"

    async def check(self, content: str) -> GuardrailFinding:
        return GuardrailFinding(triggered=False)


def test_guardrail_registry_get_given_registered_name_returns_guardrail():
    registry = GuardrailRegistry()
    guardrail = _FakeGuardrail()
    registry.register("no-secrets", guardrail)

    assert registry.get("no-secrets") is guardrail


def test_guardrail_registry_get_given_unregistered_name_raises_guardrail_not_found_error():
    registry = GuardrailRegistry()

    with pytest.raises(GuardrailNotFoundError):
        registry.get("missing")


def test_guardrail_registry_all_given_none_registered_returns_empty_dict():
    registry = GuardrailRegistry()

    assert registry.all() == {}


def test_guardrail_registry_get_many_given_names_returns_guardrails_in_order():
    registry = GuardrailRegistry()
    first = _FakeGuardrail()
    second = _FakeGuardrail()
    registry.register("first", first)
    registry.register("second", second)

    assert registry.get_many(["second", "first"]) == [second, first]


def test_guardrail_registry_get_many_given_unregistered_name_raises_guardrail_not_found_error():
    registry = GuardrailRegistry()

    with pytest.raises(GuardrailNotFoundError):
        registry.get_many(["missing"])


def test_guardrail_registry_get_many_given_empty_list_returns_empty_list():
    registry = GuardrailRegistry()

    assert registry.get_many([]) == []


def test_guardrail_finding_defaults_to_no_reason_or_redaction():
    finding = GuardrailFinding(triggered=False)

    assert finding.reason is None
    assert finding.redacted_content is None
