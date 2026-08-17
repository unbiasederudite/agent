"""Chat-completion request orchestration."""

from typing import Any

from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.protocols.itool import ITool
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.tool import ToolRegistry


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`."""
    return a if a is not None else b


def _tool_schema(tool: ITool) -> dict[str, Any]:
    """Build an OpenAI-format function schema for `tool`."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class CompletionService:
    """Orchestrates a single chat-completion request against a registered LLM.

    Always routed through a registered agent's system prompt, defaults, and tools.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        """Initialize CompletionService with LLM, agent, and tool registries.

        Args:
            llm_registry: Registry of available LLM implementations.
            agent_registry: Registry of available agent configurations.
            tool_registry: Registry of available tool implementations.
        """
        self._llm_registry = llm_registry
        self._agent_registry = agent_registry
        self._tool_registry = tool_registry

    async def run(
        self,
        messages: list[Message],
        agent: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[str] | None = None,
    ) -> Run:
        """Complete `messages`, routed through the registered agent `agent`.

        `model`, if given, overrides the agent's `default_llm`. `temperature`/`top_p`/
        `max_tokens` resolve independently as: this call's value, else the agent's
        configured value, else the LLM's own configured default (resolved inside the LLM
        implementation). The agent's `system_prompt` is unconditionally prepended as the
        leading system message -- it is never overridden by or merged with messages already
        present in `messages`.

        `tools` resolves as a tri-state, independent of the scalar params above: omitted or
        `None` uses the agent's configured `tools` (or none, if it has none); an explicit
        empty list suppresses tools entirely, even if the agent has some; a non-empty list is
        used exactly as given, ignoring the agent's own. Every resolved tool name is looked up
        in `ToolRegistry` and sent to the LLM as a function schema; nothing executes any tool
        call the LLM returns.

        Raises:
            AgentNotFoundError: if `agent` is not registered.
            LLMNotFoundError: if the resolved model is not registered.
            ToolNotFoundError: if a resolved tool name is not registered.
            LLMError: if the underlying LLM call fails.
        """
        agent_config = self._agent_registry.get(agent)
        messages = [Message(role="system", content=agent_config.system_prompt), *messages]

        effective_model = model if model is not None else agent_config.default_llm

        resolved_temperature = _first_not_none(temperature, agent_config.temperature)
        resolved_top_p = _first_not_none(top_p, agent_config.top_p)
        resolved_max_tokens = _first_not_none(max_tokens, agent_config.max_tokens)

        tool_names = tools if tools is not None else agent_config.tools
        tool_schemas = [
            _tool_schema(self._tool_registry.get(name)) for name in dict.fromkeys(tool_names)
        ] or None

        llm = self._llm_registry.get(effective_model)
        completion = await llm.complete(
            messages,
            temperature=resolved_temperature,
            top_p=resolved_top_p,
            max_tokens=resolved_max_tokens,
            tools=tool_schemas,
        )
        return Run(
            model=effective_model,
            request=messages,
            response=completion.message,
            usage=completion.usage,
            finish_reason=completion.finish_reason,
        )
