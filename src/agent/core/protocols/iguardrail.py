"""Protocol interface for guardrail implementations, and the pipeline that runs them."""

import logging
import time
from typing import Literal, Protocol

from agent.core.exceptions import GuardrailBlockedError
from agent.core.models.guardrail import GuardrailFinding

logger = logging.getLogger(__name__)


class IGuardrail(Protocol):
    """Interface for a check that inspects content and decides whether to flag it."""

    name: str  # The guardrail's lookup key, matching its config entry's `name`.
    action: Literal["block", "redact", "warn"]  # What happens when this guardrail triggers.

    async def check(self, content: str) -> GuardrailFinding:
        """Check `content` and report whether it should be flagged.

        Args:
            content: Text to check.

        Returns:
            GuardrailFinding: the check's result.
        """
        ...


async def run_guardrails(content: str, guardrails: list[IGuardrail]) -> str:
    """Run `guardrails` against `content` in order, applying each one's configured action.

    Args:
        content: Text to check.
        guardrails: Guardrails to run, in order.

    Returns:
        str: `content`, possibly redacted by one or more guardrails.

    Raises:
        GuardrailBlockedError: a block-action guardrail triggered.
    """
    for guardrail in guardrails:
        start = time.monotonic()
        finding = await guardrail.check(content)
        duration_ms = (time.monotonic() - start) * 1000
        if not finding.triggered:
            logger.debug(
                "guardrail '%s' checked in %.1fms, no findings", guardrail.name, duration_ms
            )
            continue
        if guardrail.action == "block":
            logger.info(
                "guardrail '%s' blocked this content after %.1fms: %s",
                guardrail.name,
                duration_ms,
                finding.reason,
            )
            raise GuardrailBlockedError(
                f"guardrail '{guardrail.name}' blocked this content: {finding.reason}"
            )
        if guardrail.action == "redact" and finding.redacted_content is not None:
            logger.info(
                "guardrail '%s' redacted content after %.1fms: %s",
                guardrail.name,
                duration_ms,
                finding.reason,
            )
            content = finding.redacted_content
        else:
            logger.warning(
                "guardrail '%s' triggered (%s) after %.1fms: %s",
                guardrail.name,
                guardrail.action,
                duration_ms,
                finding.reason,
            )
    return content
