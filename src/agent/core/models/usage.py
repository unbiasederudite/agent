"""Token usage model for LLM completions."""

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class Usage(BaseModel):
    """Token counts for one LLM completion."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(description="Number of tokens in the input.")
    completion_tokens: int = Field(description="Number of tokens in the generated output.")
    total_tokens: int = Field(description="Total tokens consumed (prompt + completion).")
    cost_usd: float | None = Field(
        default=None,
        description="USD cost of this completion. `None` if the model's pricing is unknown.",
    )

    @field_serializer("cost_usd", when_used="json")
    def _serialize_cost_usd(self, value: float | None) -> str | None:
        """Render as fixed-decimal text, e.g. "0.0000066", never "6.6e-06".

        Args:
            value: The cost to render.

        Returns:
            str | None: the fixed-decimal text, or `None` if `value` is `None`.
        """
        if value is None:
            return None
        text = f"{value:.10f}".rstrip("0")
        return text if not text.endswith(".") else f"{text}0"


ZERO_USAGE = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=None)


def sum_usage(a: Usage, b: Usage) -> Usage:
    """Add two Usage totals together, field by field.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Usage: the field-wise sum.
    """
    cost_usd = (
        None
        if a.cost_usd is None and b.cost_usd is None
        else round((a.cost_usd or 0) + (b.cost_usd or 0), 10)
    )
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        cost_usd=cost_usd,
    )
