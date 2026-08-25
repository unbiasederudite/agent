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


class LLMError(AgentError):
    """Raised when an outbound LLM call fails."""


class LLMRateLimitedError(LLMError):
    """Raised when the upstream LLM provider rate-limited the request."""


class LLMTimeoutError(LLMError):
    """Raised when the upstream LLM provider request timed out."""
