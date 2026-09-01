"""The full exception hierarchy, rooted at AgentError."""


class AgentError(Exception):
    """Root of the agent-core exception hierarchy."""


class ConfigError(AgentError):
    """Raised when startup configuration is invalid."""


class LLMNotFoundError(AgentError):
    """Raised when a requested LLM name is not registered."""


class AgentNotFoundError(AgentError):
    """Raised when a requested agent name is not registered."""


class ToolNotFoundError(AgentError):
    """Raised when a requested tool name is not registered."""


class StrategyNotFoundError(AgentError):
    """Raised when a requested reasoning strategy name is not registered."""


class SessionNotFoundError(AgentError):
    """Raised when a requested (agent, session_id) pair is not registered.

    Also raised when session_id is valid but was created under a different agent --
    a session is locked to its creating agent, so cross-agent reuse is indistinguishable
    from an unknown session.
    """


class LLMError(AgentError):
    """Raised when an outbound LLM call fails."""


class LLMRateLimitedError(LLMError):
    """Raised when the upstream LLM provider rate-limited the request."""


class LLMTimeoutError(LLMError):
    """Raised when the upstream LLM provider request timed out."""


class LLMOverloadedError(LLMError):
    """Raised when a model's configured `max_concurrent_requests` is already reached.

    Checked before any provider call is attempted -- never itself retried by the adapter's
    own retry loop, since retrying it internally would quietly reintroduce a wait, contradicting
    the immediate-rejection behavior this exists to provide.
    """


class LLMContextWindowExceededError(LLMError):
    """Raised when the provider rejects a request for exceeding the model's context window."""


class CompactionExhaustedError(LLMContextWindowExceededError):
    """Raised when compaction was tried and the request still doesn't fit.

    The single compact-and-retry attempt -- either compaction itself failed to produce a
    usable summary, or the retried request still overflowed -- still left the request too
    big for the model's context window. `CompactionConfig.keep_recent_turns` is never
    overridden to recover from this: the turns it protects are never summarized away under
    any circumstance, so a session whose protected window alone doesn't fit is a genuine,
    permanent dead end for that session/model combination until the model or config changes.

    Distinguishes "compaction was tried and could not help" (this) from "compaction wasn't
    available to try at all" (`LLMContextWindowExceededError` on its own -- e.g. no
    `compaction_service` configured, or no prior `session_id`) -- both are still overflow
    errors, but only this one indicates a likely session/config problem worth diagnosing.
    """


class InputTooLargeError(AgentError):
    """Raised when a request's `message` exceeds the agent's configured `max_input_chars`."""


class SessionBusyError(AgentError):
    """Raised when a request targets a session another operation is currently using.

    Two requests against the same (agent, session_id) reading the starting history before
    either has written anything produces a well-formed but semantically confusing result --
    the second request's answer never sees the first's exchange, even though the stored
    transcript reads as an ordinary sequential conversation on replay. Rejected outright
    instead, matching how OpenAI's own Assistants API handles a second run against a
    thread that already has one active.
    """


class RequestTimeoutError(AgentError):
    """Raised when a single request exceeds its agent's configured `max_request_seconds`.

    A single overall budget across everything one call to `AgentRunService.run()` might
    do -- adapter retries and their backoff, every tool-calling round, a reactive
    compaction-and-retry pass, and any proactive compaction the call triggers -- none of
    which bounds the others stacking up together. Enforced via a single
    `asyncio.wait_for()` wrapper, not threaded through each inner layer individually.

    The session's stored history may already have been rewritten by a compaction pass
    that completed before the timeout fired -- compaction's own `replace()` call commits
    immediately on success, so a timeout during the LLM call that follows it does not undo
    that commit. The caller receives this error with no indication the rewrite happened;
    a retry continues from the already-compacted history, not the original.
    """


class ToolNotAllowedError(AgentError):
    """Raised when a request names a tool that exists but isn't permitted for this agent.

    Distinct from `ToolNotFoundError`, which means the name isn't registered anywhere in
    the process at all -- this means it exists, but falls outside this agent's own
    configured `allowed_tools` ceiling.
    """


class ModelNotAllowedError(AgentError):
    """Raised when a request names a model that exists but isn't permitted for this agent.

    Distinct from `LLMNotFoundError` (not registered at all) -- this means it exists, but
    falls outside this agent's own configured `allowed_models` ceiling.
    """


class StrategyNotAllowedError(AgentError):
    """Raised when a request names a strategy that exists but isn't permitted for this agent.

    Distinct from `StrategyNotFoundError` (not registered at all) -- this means it exists,
    but falls outside this agent's own configured `allowed_strategies` ceiling.
    """
