import pytest

from agent.core.exceptions import AgentNotFoundError
from agent.core.models.config import AgentConfig
from agent.core.registries.agent import AgentRegistry


def test_agent_registry_get_given_registered_name_returns_agent():
    registry = AgentRegistry()
    agent = AgentConfig(
        name="researcher", system_prompt="You are helpful.", default_llm="openai/gpt-4o"
    )
    registry.register("researcher", agent)

    assert registry.get("researcher") is agent


def test_agent_registry_get_given_unregistered_name_raises_agent_not_found_error():
    registry = AgentRegistry()

    with pytest.raises(AgentNotFoundError):
        registry.get("missing")
