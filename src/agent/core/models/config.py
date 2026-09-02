"""Pydantic config models for application startup."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SamplingDefaults(BaseModel):
    """Sampling-default fields shared by `LLMConfig` and `AgentConfig`."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Default sampling temperature, if set. Only the lower bound is enforced here "
            "-- the upper bound is provider-specific (2 for OpenAI, 1 for Anthropic), so a "
            "too-high value is left for the provider itself to reject, matching "
            "`AgentRunRequest.temperature` in `api/schemas.py`."
        ),
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Default nucleus sampling value, if set."
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Default max output tokens, if set."
    )


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
    num_retries: int = Field(
        default=2,
        ge=0,
        description="Retries for retriable failures (rate limits, timeouts, connection errors, "
        "5xx) before giving up. Permanent errors (bad request, auth, not found, etc.) are never "
        "retried, regardless of this value.",
    )
    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Per-attempt timeout in seconds. None leaves litellm's own default in "
        "place (120s via LITELLM_TIMEOUT) -- this is a per-model override, not a fallback for "
        "an otherwise-unbounded call.",
    )
    retry_base_delay: float = Field(
        default=1.0, gt=0, description="Delay before the first retry, in seconds."
    )
    retry_max_delay: float = Field(
        default=30.0, gt=0, description="Cap on delay between retries, in seconds."
    )
    retry_multiplier: float = Field(
        default=2.0, ge=1, description="Backoff multiplier applied to the delay after each retry."
    )
    max_concurrent_requests: int | None = Field(
        default=None,
        ge=1,
        description="Cap on concurrent in-flight calls to this model. None means unlimited. A "
        "request that finds the cap already reached is rejected immediately with "
        "LLMOverloadedError, never queued.",
    )

    @model_validator(mode="after")
    def _max_delay_not_below_base(self) -> "LLMConfig":
        """Reject a max delay tighter than the base delay.

        Every retry would silently collapse to the max, contradicting the configured
        base delay.
        """
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError(
                f"retry_max_delay ({self.retry_max_delay}) must be >= "
                f"retry_base_delay ({self.retry_base_delay})"
            )
        return self


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level to emit."
    )
    format: Literal["text", "json"] = Field(
        default="text",
        description="Text is a single formatted line; json renders each record (and any extra "
        "structured fields a call site attached) as one JSON object per line.",
    )
    console: bool = Field(default=True, description="Whether to log to the console (stderr).")
    file: str | None = Field(
        default=None, description="Path to a log file, or None to skip file output."
    )
    file_max_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Rotate once the file reaches this size, in bytes. None means an uncapped "
        "single file -- no rotation. Ignored if `file` is unset.",
    )
    file_backup_count: int = Field(
        default=5,
        ge=0,
        description="Rotated backups to keep. Only relevant when file_max_bytes is set.",
    )

    @model_validator(mode="after")
    def _require_a_destination(self) -> "LoggingConfig":
        """Reject console=False with no file -- that combination produces zero log output."""
        if not self.console and self.file is None:
            raise ValueError(
                "logging.console is False but logging.file is not set -- "
                "no log output would ever be produced"
            )
        return self


class ToolConfig(BaseModel):
    """One entry in the startup tool allow-list.

    `name` must match a code-level tool implementation (see `core/factories/app.py`)
    and is also the ToolRegistry lookup key.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The tool's lookup key, matching a code-level implementation.")


class StrategyConfig(BaseModel):
    """One entry in the startup strategy allow-list.

    `name` must match a code-level strategy implementation (see `core/factories/app.py`)
    and is also the StrategyRegistry lookup key.
    """

    model_config = ConfigDict(extra="forbid")

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
    allowed_tools: list[str] | None = Field(
        default=None,
        description=(
            "Ceiling on which tool names a request may ever specify for this agent, "
            "including as an override -- `None` means unrestricted (any registered tool "
            "may be requested, current behavior); `[]` means this agent may never use "
            "tools no matter what a request asks for; a non-empty list means only these "
            "names are ever permitted, as defaults or as overrides. `tools` must already "
            "be within this ceiling when both are set."
        ),
    )
    allowed_models: list[str] | None = Field(
        default=None,
        description=(
            "Ceiling on which model names a request may override to for this agent -- "
            "`None` means unrestricted (current behavior); a non-empty list means only "
            "these models are ever permitted. No empty-list state (an agent needs some "
            "model to function). `model` must already be within this ceiling when set."
        ),
    )
    allowed_strategies: list[str] | None = Field(
        default=None,
        description=(
            "Ceiling on which strategy names a request may override to for this agent -- "
            "same shape and reasoning as `allowed_models`. Primarily a guard against "
            "exposing a new or externally-sourced strategy implementation (e.g. one "
            "wrapping an external framework) to every agent the moment it's registered, "
            "before its IStrategy contract compliance has been verified in practice."
        ),
    )
    max_request_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Overall wall-clock budget for one call to AgentRunService.run(), covering "
            "every adapter retry/backoff, every tool-calling round, and any reactive "
            "compaction-and-retry pass together -- `None` means unbounded (current "
            "behavior). Enforced by a single asyncio.wait_for() wrapper around the whole "
            "call, not threaded through each inner layer."
        ),
    )

    @model_validator(mode="after")
    def _defaults_within_ceilings(self) -> "AgentConfig":
        """Reject a default that falls outside this agent's own configured ceiling."""
        if self.allowed_tools is not None:
            outside = [name for name in self.tools if name not in self.allowed_tools]
            if outside:
                raise ValueError(f"tools {outside} are not in allowed_tools {self.allowed_tools}")
        if self.allowed_models is not None and self.model not in self.allowed_models:
            raise ValueError(f"model '{self.model}' is not in allowed_models {self.allowed_models}")
        if self.allowed_strategies is not None and self.strategy not in self.allowed_strategies:
            raise ValueError(
                f"strategy '{self.strategy}' is not in allowed_strategies {self.allowed_strategies}"
            )
        return self


class CompactionConfig(BaseModel):
    """Global settings for summarizing a session's older history once it crosses budget."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
    max_sessions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Caps how many distinct (agent, session_id) sessions InMemorySessionStore, "
            "CostTracker, and ContextFootprintTracker each keep at once. `None` means "
            "unbounded (current behavior). Once over the cap, the least-recently-touched "
            "session is evicted -- InMemorySessionStore's stored history and lock "
            "together, CostTracker's cumulative-usage entries, and "
            "ContextFootprintTracker's context-footprint entries (the single source of "
            "truth CompactionService reads for a session's current context size -- "
            "CompactionService keeps no state of its own here, and neither it nor "
            "CostTracker depends on the other), each independently -- these are three "
            "uncoordinated LRUs sharing the same configured limit, not one coordinated "
            "eviction event; either can evict a session another still holds, and that's "
            "harmless (a missing footprint just skips one proactive compaction check and "
            "reads as an unknown context size on the usage endpoint; a missing session "
            "just reads as not-found; missing usage state reads as never-recorded). A "
            "session actively in use (its lock held, or marked busy) is never evicted "
            "from InMemorySessionStore -- eviction explicitly skips any such entry, "
            "never relying merely on it typically being the most recently touched. "
            "CostTracker's and ContextFootprintTracker's eviction has no such check -- "
            "each evicts purely by LRU recency, even for a session that is currently busy."
        ),
    )
