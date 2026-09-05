"""Guardrail check result model."""

from pydantic import BaseModel, Field


class GuardrailFinding(BaseModel):
    """Result of one guardrail check against a piece of content."""

    triggered: bool = Field(description="Whether this check flagged the content.")
    reason: str | None = Field(
        default=None, description="Human-readable explanation of why the check triggered."
    )
    redacted_content: str | None = Field(
        default=None,
        description="Corrected content, when the check triggered and can produce one.",
    )
