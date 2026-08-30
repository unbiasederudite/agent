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
