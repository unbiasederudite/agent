import logging

import pytest

from agent.core.exceptions import GuardrailBlockedError
from agent.core.models.guardrail import GuardrailFinding
from agent.core.protocols.iguardrail import run_guardrails


class _Guardrail:
    def __init__(self, name: str, action: str, finding: GuardrailFinding) -> None:
        self.name = name
        self.action = action
        self._finding = finding
        self.checked_with: list[str] = []

    async def check(self, content: str) -> GuardrailFinding:
        self.checked_with.append(content)
        return self._finding


async def test_run_guardrails_given_nothing_triggers_returns_content_unchanged():
    guardrail = _Guardrail("g", "block", GuardrailFinding(triggered=False))

    result = await run_guardrails("hello", [guardrail])

    assert result == "hello"


async def test_run_guardrails_given_nothing_triggers_logs_the_passed_check(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail("no-secrets", "block", GuardrailFinding(triggered=False))

    with caplog.at_level(logging.DEBUG):
        await run_guardrails("hello", [guardrail])

    assert "no-secrets" in caplog.text


async def test_run_guardrails_given_block_action_triggers_raises_guardrail_blocked_error():
    guardrail = _Guardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="looks like a key")
    )

    with pytest.raises(GuardrailBlockedError, match="no-secrets"):
        await run_guardrails("sk-abc123", [guardrail])


async def test_run_guardrails_given_block_action_triggers_logs_before_raising(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail(
        "no-secrets", "block", GuardrailFinding(triggered=True, reason="looks like a key")
    )

    with caplog.at_level(logging.INFO), pytest.raises(GuardrailBlockedError):
        await run_guardrails("sk-abc123", [guardrail])

    assert "no-secrets" in caplog.text


async def test_run_guardrails_given_redact_action_triggers_returns_redacted_content():
    guardrail = _Guardrail(
        "no-pii",
        "redact",
        GuardrailFinding(triggered=True, reason="pii", redacted_content="[REDACTED]"),
    )

    result = await run_guardrails("my email is a@b.com", [guardrail])

    assert result == "[REDACTED]"


async def test_run_guardrails_given_redact_action_triggers_logs_the_redaction(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail(
        "no-pii",
        "redact",
        GuardrailFinding(triggered=True, reason="pii", redacted_content="[REDACTED]"),
    )

    with caplog.at_level(logging.INFO):
        await run_guardrails("my email is a@b.com", [guardrail])

    assert "no-pii" in caplog.text


async def test_run_guardrails_given_redact_action_with_no_fix_returns_content_unchanged():
    guardrail = _Guardrail("no-pii", "redact", GuardrailFinding(triggered=True, reason="pii"))

    result = await run_guardrails("content", [guardrail])

    assert result == "content"


async def test_run_guardrails_given_warn_action_triggers_logs_and_returns_content_unchanged(
    caplog: pytest.LogCaptureFixture,
):
    guardrail = _Guardrail(
        "toxicity", "warn", GuardrailFinding(triggered=True, reason="mildly rude")
    )

    with caplog.at_level(logging.WARNING):
        result = await run_guardrails("content", [guardrail])

    assert result == "content"
    assert "toxicity" in caplog.text


async def test_run_guardrails_given_multiple_guardrails_feeds_redacted_content_forward():
    first = _Guardrail(
        "first",
        "redact",
        GuardrailFinding(triggered=True, reason="a", redacted_content="stage-1"),
    )
    second = _Guardrail("second", "block", GuardrailFinding(triggered=False))

    result = await run_guardrails("original", [first, second])

    assert result == "stage-1"
    assert second.checked_with == ["stage-1"]


async def test_run_guardrails_given_empty_list_returns_content_unchanged():
    result = await run_guardrails("content", [])

    assert result == "content"


async def test_run_guardrails_given_warn_then_block_still_evaluates_and_raises_on_the_second():
    first = _Guardrail("first", "warn", GuardrailFinding(triggered=True, reason="mild"))
    second = _Guardrail("second", "block", GuardrailFinding(triggered=True, reason="severe"))

    with pytest.raises(GuardrailBlockedError, match="second"):
        await run_guardrails("original", [first, second])

    assert second.checked_with == ["original"]
