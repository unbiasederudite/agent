"""Pydantic config models for application startup."""

from typing import Literal

from pydantic import BaseModel, Field


class SamplingDefaults(BaseModel):
    """Sampling-default fields shared by `LLMConfig` and `AgentConfig`."""

    temperature: float | None = Field(
        default=None, description="Default sampling temperature, if set."
    )
    top_p: float | None = Field(default=None, description="Default nucleus sampling value, if set.")
    max_tokens: int | None = Field(default=None, description="Default max output tokens, if set.")


class LLMConfig(SamplingDefaults):
    """One entry in the startup LLM allow-list.

    `model` is a litellm-format provider/model id (e.g. "anthropic/claude-sonnet-5")
    and doubles as the LLMRegistry lookup key.
    """

    model: str = Field(
        description='litellm-format provider/model id, e.g. "anthropic/claude-sonnet-5".'
    )
    context_window: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Overrides litellm's own context-window lookup for this model, used by "
            "CompactionService's budget check. Set this for models litellm's static "
            "data doesn't recognize (self-hosted endpoints, fine-tuned model ids) -- "
            "when set, it's used directly, unconditionally, with no litellm lookup at "
            "all for this model, even for models litellm would otherwise recognize "
            "correctly. Leave unset to let it be looked up from litellm automatically."
        ),
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level to emit."
    )


class ToolConfig(BaseModel):
    """One entry in the startup tool allow-list.

    `name` must match a code-level tool implementation (see `core/factories/app.py`)
    and is also the ToolRegistry lookup key.
    """

    name: str = Field(description="The tool's lookup key, matching a code-level implementation.")


class StrategyConfig(BaseModel):
    """One entry in the startup strategy allow-list.

    `name` must match a code-level strategy implementation (see `core/factories/app.py`)
    and is also the StrategyRegistry lookup key.
    """

    name: str = Field(
        description="The strategy's lookup key, matching a code-level implementation."
    )


class AgentConfig(SamplingDefaults):
    """One entry in the startup agent allow-list.

    `name` is the AgentRegistry lookup key. `system_prompt` is unconditionally prepended
    as the leading system message whenever this agent is selected -- it is identity
    content, not a request-overridable value. `model` must match a declared
    `LLMConfig.model` in the same config. Every name in `tools` must match a declared
    `ToolConfig.name` in the same config.
    """

    name: str = Field(description="The agent's lookup key in the AgentRegistry.")
    system_prompt: str = Field(
        description=(
            "Unconditionally prepended as the leading system message when this agent "
            "is selected; never overridden by client-supplied messages."
        )
    )
    model: str = Field(
        description="The LLMConfig.model used when the request doesn't override `model`."
    )
    strategy: str = Field(
        description=(
            "The StrategyRegistry lookup key used when the request doesn't override `strategy`."
        )
    )
    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names available to this agent by default, unless the request "
            "overrides them (see AgentRunService.run's tri-state `tools` resolution)."
        ),
    )
    max_tool_iterations: int = Field(
        default=10,
        ge=1,
        description="Caps rounds of call-LLM/maybe-call-a-tool a tool-calling strategy may run.",
    )
    max_tool_result_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Caps the length of a single tool call's result content; a longer result is "
            "truncated with a trailing marker noting how much was cut. `None` means "
            "uncapped. Protects a single run's context from a tool returning unbounded "
            "content (e.g. reading a large file) -- independent of session-level "
            "compaction, which only ever operates on already-stored history between "
            "turns, never on what's happening inside one in-progress run."
        ),
    )
    max_tool_calls_per_round: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Caps how many tool calls from a single LLM response actually execute; a "
            "response requesting more has the excess calls skipped, each replaced with a "
            "short error result noting the limit -- the LLM sees this and can adjust on "
            "its next round rather than the loop silently dropping them. `None` means "
            "uncapped. A single response can request arbitrarily many simultaneous tool "
            "calls otherwise, each contributing its own result content with no ceiling on "
            "how many pile up in one round."
        ),
    )
    max_tool_results_total_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Caps the combined length of every tool result's content across a whole run "
            "(all rounds, not just one), independent of `max_tool_result_chars`'s per-call "
            "cap -- many individually-small results can still add up. Once the running "
            "total reaches this budget, every further tool result is replaced with a "
            "short marker instead of its real content; `None` means uncapped. Not a hard "
            "ceiling on its own: the one result that crosses the threshold is still "
            "admitted in full before the marker kicks in for anything after it -- pair "
            "this with `max_tool_result_chars` for an actual per-result size guarantee too."
        ),
    )
    max_input_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Caps the length of a request's `message` for this agent; `None` means "
            "uncapped. Per-agent, not global -- different agents have different natural "
            "input sizes."
        ),
    )


class CompactionConfig(BaseModel):
    """Global settings for summarizing a session's older history once it crosses budget."""

    model: str = Field(
        description=(
            "LLMConfig.model used to generate summaries -- typically a cheaper model. Its "
            "own LLMConfig.max_tokens should be set generously: a summary's length doesn't "
            "scale with how much history it's summarizing (a good summary of a long session "
            "is still short), so a low max_tokens risks the summary itself getting "
            'truncated (see CompactionService.compact\'s finish_reason == "length" '
            "handling) -- not a correctness bug once that guard exists, but a session "
            "stuck unable to compact, repeatedly, until the config is fixed."
        )
    )
    token_budget_pct: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description=(
            "Fraction of the resolved model's max_input_tokens (see ILLM.max_input_tokens) "
            "that triggers compaction. Leave headroom below 1.0: this is checked before a "
            "turn starts, using the *previous* turn's ending size (see CompactionService) -- "
            "it can't stop the turn in progress from overshooting, only the next one from "
            "starting oversized."
        ),
    )
    keep_recent_turns: int = Field(
        default=4,
        ge=0,
        description=(
            'Most recent complete turns (each starting at one role="user" message) kept '
            "verbatim, never summarized. Counted in turns, not raw messages, so a tool-call/"
            "tool-result pair is never split across the summarized/kept boundary."
        ),
    )
    chunk_turns: int = Field(
        default=4,
        ge=1,
        description=(
            "Turns per chunk in the map-reduce fallback used when a single-pass summary "
            "overflows the summarizer's own context window -- independent of "
            "keep_recent_turns (a different concept: turns kept verbatim, never "
            "summarized at all)."
        ),
    )
    prompt: str = Field(
        default=(
            "Summarize the conversation above concisely. Preserve the task's goal, key "
            "decisions, established constraints, and any facts or results later turns may "
            "need to reference. Omit small talk and resolved intermediate steps."
        ),
        description=(
            "Appended as a final user-role message after the old messages being summarized."
        ),
    )


class AppConfig(BaseModel):
    """Root startup configuration, loaded once from a JSON file."""

    llms: list[LLMConfig] = Field(description="The allow-list of LLMs available to this process.")
    agents: list[AgentConfig] = Field(
        default_factory=list, description="The allow-list of agents available to this process."
    )
    tools: list[ToolConfig] = Field(
        default_factory=list, description="The allow-list of tools available to this process."
    )
    strategies: list[StrategyConfig] = Field(
        default_factory=list,
        description="The allow-list of reasoning strategies available to this process.",
    )
    base_prompt: str | None = Field(
        default=None,
        description=(
            "Prepended before every agent's own `system_prompt`, merged into the same "
            "leading system message -- not a second one. Shared instructions across every "
            "agent in this deployment (e.g. house style, compliance rules); omit if there's "
            "nothing to share."
        ),
    )
    compaction: CompactionConfig | None = Field(
        default=None,
        description=(
            "Global settings for summarizing a session's older history once it crosses a "
            "token budget. Omitted entirely disables compaction: no proactive check, no "
            "reactive fallback, sessions behave as if this milestone didn't exist."
        ),
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration."
    )
