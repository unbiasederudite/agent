"""Chat-completion request orchestration."""

from agent.core.exceptions import AgentError
from agent.core.models.config import AgentConfig
from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry


def _first_not_none[T](a: T | None, b: T | None) -> T | None:
    """Return `a`, or `b` if `a` is `None`."""
    return a if a is not None else b


class CompletionService:
    """Orchestrates a single chat-completion request against a registered LLM.

    Optionally routed through a registered agent's system prompt and defaults.
    """

    def __init__(self, llm_registry: LLMRegistry, agent_registry: AgentRegistry) -> None:
        """Initialize CompletionService with LLM and agent registries.

        Args:
            llm_registry: Registry of available LLM implementations.
            agent_registry: Registry of available agent configurations.
        """
        self._llm_registry = llm_registry
        self._agent_registry = agent_registry

    async def run(
        self,
        messages: list[Message],
        *,
        agent: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Run:
        """Complete `messages`, optionally routed through a registered agent.

        `model`, if given, always selects the LLM to use; otherwise the selected agent's
        `default_llm` is used. `temperature`/`top_p`/`max_tokens` resolve independently as:
        this call's value, else the agent's configured value, else the LLM's own configured
        default (resolved inside the LLM implementation). The agent's `system_prompt`, if an
        agent is selected, is unconditionally prepended as the leading system message -- it
        is never overridden by or merged with messages already present in `messages`.

        Raises:
            AgentError: if neither `agent` nor `model` is given.
            AgentNotFoundError: if `agent` is given and not registered.
            LLMNotFoundError: if the resolved model is not registered.
            LLMError: if the underlying LLM call fails.
        """
        agent_config: AgentConfig | None = None
        if agent is not None:
            agent_config = self._agent_registry.get(agent)
            messages = [Message(role="system", content=agent_config.system_prompt), *messages]

        if model is not None:
            effective_model = model
        elif agent_config is not None:
            effective_model = agent_config.default_llm
        else:
            raise AgentError("either `agent` or `model` must be given")

        resolved_temperature = _first_not_none(
            temperature, agent_config.temperature if agent_config is not None else None
        )
        resolved_top_p = _first_not_none(
            top_p, agent_config.top_p if agent_config is not None else None
        )
        resolved_max_tokens = _first_not_none(
            max_tokens, agent_config.max_tokens if agent_config is not None else None
        )

        llm = self._llm_registry.get(effective_model)
        completion = await llm.complete(
            messages,
            temperature=resolved_temperature,
            top_p=resolved_top_p,
            max_tokens=resolved_max_tokens,
        )
        return Run(
            model=effective_model,
            request=messages,
            response=completion.message,
            usage=completion.usage,
            finish_reason=completion.finish_reason,
        )
