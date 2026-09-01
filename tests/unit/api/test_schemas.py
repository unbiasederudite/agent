"""Tests for api/schemas.py -- request/response field constraints."""

import pytest
from pydantic import ValidationError

from agent.api.schemas import AgentRunRequest


def test_agent_run_request_given_valid_temperature_constructs():
    request = AgentRunRequest(message="hi", temperature=1.0)

    assert request.temperature == 1.0


def test_agent_run_request_given_negative_temperature_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentRunRequest(message="hi", temperature=-0.1)


def test_agent_run_request_given_temperature_above_two_still_constructs():
    # Deliberately no upper bound: the ceiling is provider-specific (2 for OpenAI, 1 for
    # Anthropic), so a too-high value is left for the provider to reject rather than
    # guessed at here and potentially validated wrong for the agent's actual model.
    request = AgentRunRequest(message="hi", temperature=2.1)

    assert request.temperature == 2.1


def test_agent_run_request_given_valid_top_p_constructs():
    request = AgentRunRequest(message="hi", top_p=0.9)

    assert request.top_p == 0.9


def test_agent_run_request_given_negative_top_p_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentRunRequest(message="hi", top_p=-0.1)


def test_agent_run_request_given_top_p_above_one_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentRunRequest(message="hi", top_p=1.1)


def test_agent_run_request_given_no_temperature_or_top_p_constructs():
    request = AgentRunRequest(message="hi")

    assert request.temperature is None
    assert request.top_p is None


def test_agent_run_request_given_valid_max_tokens_constructs():
    request = AgentRunRequest(message="hi", max_tokens=100)

    assert request.max_tokens == 100


def test_agent_run_request_given_zero_max_tokens_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentRunRequest(message="hi", max_tokens=0)


def test_agent_run_request_given_negative_max_tokens_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentRunRequest(message="hi", max_tokens=-1)


def test_agent_run_request_given_unknown_field_raises_validation_error():
    with pytest.raises(ValidationError):
        AgentRunRequest(message="hi", sesion_id="typo")
