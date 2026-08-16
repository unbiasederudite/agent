"""Token usage model for LLM completions."""

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token counts for one LLM completion."""

    prompt_tokens: int = Field(description="Number of tokens in the input.")
    completion_tokens: int = Field(description="Number of tokens in the generated output.")
    total_tokens: int = Field(description="Total tokens consumed (prompt + completion).")
