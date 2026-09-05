from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    ConfigError,
    LLMError,
    LLMNotFoundError,
    LLMOverloadedError,
    LLMRateLimitedError,
    LLMTimeoutError,
    ModelNotAllowedError,
    RequestTimeoutError,
    SessionBusyError,
    SessionNotFoundError,
    StrategyNotAllowedError,
    StrategyNotFoundError,
    ToolNotAllowedError,
    ToolNotFoundError,
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


def test_tool_not_found_error_is_agent_error():
    assert issubclass(ToolNotFoundError, AgentError)


def test_strategy_not_found_error_is_agent_error():
    assert issubclass(StrategyNotFoundError, AgentError)


def test_session_not_found_error_is_agent_error():
    assert issubclass(SessionNotFoundError, AgentError)


def test_llm_overloaded_error_is_llm_error():
    assert issubclass(LLMOverloadedError, LLMError)


def test_session_busy_error_is_agent_error():
    assert issubclass(SessionBusyError, AgentError)


def test_request_timeout_error_is_agent_error():
    assert issubclass(RequestTimeoutError, AgentError)


def test_tool_not_allowed_error_is_agent_error():
    assert issubclass(ToolNotAllowedError, AgentError)


def test_model_not_allowed_error_is_agent_error():
    assert issubclass(ModelNotAllowedError, AgentError)


def test_strategy_not_allowed_error_is_agent_error():
    assert issubclass(StrategyNotAllowedError, AgentError)
