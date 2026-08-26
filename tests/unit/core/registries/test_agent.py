import pytest

from agent.core.exceptions import AgentNotFoundError
from agent.core.models.config import AgentConfig
from agent.core.registries.agent import AgentRegistry


def test_agent_registry_get_given_registered_name_returns_agent():
    registry = AgentRegistry()
    agent = AgentConfig(
        name="researcher",
        system_prompt="You are helpful.",
        model="openai/gpt-4o",
        strategy="react",
    )
    registry.register("researcher", agent)

    assert registry.get("researcher") is agent


def test_agent_registry_get_given_unregistered_name_raises_agent_not_found_error():
    registry = AgentRegistry()

    with pytest.raises(AgentNotFoundError):
        registry.get("missing")


def test_agent_registry_all_given_registered_agents_returns_name_to_instance_mapping():
    registry = AgentRegistry()
    agent = AgentConfig(
        name="researcher",
        system_prompt="You are helpful.",
        model="openai/gpt-4o",
        strategy="react",
    )
    registry.register("researcher", agent)

    assert registry.all() == {"researcher": agent}


def test_agent_registry_all_given_none_registered_returns_empty_dict():
    registry = AgentRegistry()

    assert registry.all() == {}
