"""Chat-completion request orchestration."""

from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.registries.llm import LLMRegistry


class CompletionService:
    """Orchestrates a single chat-completion request against a registered LLM."""

    def __init__(self, llm_registry: LLMRegistry) -> None:
        """Initialize CompletionService with an LLM registry.

        Args:
            llm_registry: Registry of available LLM implementations.
        """
        self._llm_registry = llm_registry

    async def run(self, model: str, messages: list[Message]) -> Run:
        """Complete `messages` against the LLM registered as `model`.

        Raises:
            LLMNotFoundError: if `model` is not registered.
            LLMError: if the underlying LLM call fails.
        """
        llm = self._llm_registry.get(model)
        completion = await llm.complete(messages)
        return Run(
            model=model,
            request=messages,
            response=completion.message,
            usage=completion.usage,
            finish_reason=completion.finish_reason,
        )
