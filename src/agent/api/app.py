"""FastAPI application with OpenAI-compatible chat completions route."""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException

from agent.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from agent.core.exceptions import (
    AgentError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from agent.core.factories.app import build_llm_registry
from agent.core.models.message import Message
from agent.core.services.completion import CompletionService


def add_chat_completions_route(app: FastAPI, completion_service: CompletionService) -> None:
    """Register POST /v1/chat/completions on `app`, backed by `completion_service`."""

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        messages = [Message(role=m.role, content=m.content) for m in request.messages]
        try:
            run = await completion_service.run(request.model, messages)
        except LLMNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LLMRateLimitedError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except LLMTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except AgentError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=run.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=run.response.role, content=run.response.content),
                    finish_reason=run.finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=run.usage.prompt_tokens,
                completion_tokens=run.usage.completion_tokens,
                total_tokens=run.usage.total_tokens,
            ),
        )


def create_app(config_path: Path) -> FastAPI:
    """Build the FastAPI app, wired from the AppConfig JSON at `config_path`."""
    llm_registry = build_llm_registry(config_path)
    completion_service = CompletionService(llm_registry)

    app = FastAPI()
    add_chat_completions_route(app, completion_service)
    return app
