from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.app import add_registry_routes
from agent.core.models.completion import Completion
from agent.core.models.config import AgentConfig
from agent.core.models.message import Message
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.tools.get_current_time import GetCurrentTimeTool


class _FakeLLM:
    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> Completion:
        raise NotImplementedError


def _client() -> TestClient:
    agent_registry = AgentRegistry()
    agent_registry.register(
        "researcher",
        AgentConfig(
            name="researcher",
            system_prompt="be helpful",
            default_llm="openai/gpt-4o",
            tools=["get_current_time"],
        ),
    )
    tool_registry = ToolRegistry()
    tool_registry.register("get_current_time", GetCurrentTimeTool())
    llm_registry = LLMRegistry()
    llm_registry.register("openai/gpt-4o", _FakeLLM())

    app = FastAPI()
    add_registry_routes(app, agent_registry, tool_registry, llm_registry)
    return TestClient(app)


def test_list_agents_returns_registered_agents():
    response = _client().get("/v1/agents")

    assert response.status_code == 200
    assert response.json() == [
        {"name": "researcher", "default_llm": "openai/gpt-4o", "tools": ["get_current_time"]}
    ]


def test_list_agents_given_none_registered_returns_empty_list():
    app = FastAPI()
    add_registry_routes(app, AgentRegistry(), ToolRegistry(), LLMRegistry())

    response = TestClient(app).get("/v1/agents")

    assert response.status_code == 200
    assert response.json() == []


def test_list_tools_returns_registered_tools_with_schema():
    response = _client().get("/v1/tools")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["name"] == "get_current_time"
    assert body[0]["description"]
    assert body[0]["parameters"] == {"type": "object", "properties": {}}


def test_list_llms_returns_registered_model_names():
    response = _client().get("/v1/llms")

    assert response.status_code == 200
    assert response.json() == ["openai/gpt-4o"]
