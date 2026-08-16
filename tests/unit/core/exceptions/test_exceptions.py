from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    ConfigError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitedError,
    LLMTimeoutError,
)


def test_agent_not_found_error_is_agent_error():
    assert issubclass(AgentNotFoundError, AgentError)


def test_config_error_is_agent_error():
    assert issubclass(ConfigError, AgentError)


def test_llm_not_found_error_is_agent_error():
    assert issubclass(LLMNotFoundError, AgentError)


def test_llm_error_is_agent_error():
    assert issubclass(LLMError, AgentError)


def test_llm_rate_limited_error_is_llm_error():
    assert issubclass(LLMRateLimitedError, LLMError)


def test_llm_timeout_error_is_llm_error():
    assert issubclass(LLMTimeoutError, LLMError)
