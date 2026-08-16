"""FastAPI application with OpenAI-compatible chat completions route."""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ErrorDetail,
    ErrorResponse,
)
from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitedError,
    LLMTimeoutError,
)
from agent.core.factories.app import build_registries
from agent.core.models.message import Message
from agent.core.services.completion import CompletionService

# Maps an HTTP status code to its OpenAI-compatible (type, default code). Used whenever the
# raising site didn't attach its own `code` (see `_error_response`).
_STATUS_TO_TYPE_AND_CODE: dict[int, tuple[str, str | None]] = {
    400: ("invalid_request_error", None),
    404: ("invalid_request_error", None),
    429: ("rate_limit_error", "rate_limit_exceeded"),
    502: ("api_error", None),
    504: ("api_error", "timeout"),
    500: ("api_error", None),
}


def _error_response(
    status_code: int, message: str, param: str | None = None, code: str | None = None
) -> JSONResponse:
    """Build an OpenAI-compatible `{"error": {...}}` JSON response."""
    error_type, default_code = _STATUS_TO_TYPE_AND_CODE.get(status_code, ("api_error", None))
    body = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code if code is not None else default_code,
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def add_chat_completions_route(app: FastAPI, completion_service: CompletionService) -> None:
    """Register POST /v1/chat/completions on `app`, backed by `completion_service`."""

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0]
        loc = [str(p) for p in first["loc"] if p != "body"]
        message = first["msg"].removeprefix("Value error, ")
        return _error_response(400, message, param=".".join(loc) if loc else None)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            message = str(exc.detail["message"])
            code = exc.detail.get("code")
        else:
            message = str(exc.detail)
            code = None
        return _error_response(exc.status_code, message, code=code)

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_request: Request, exc: Exception) -> JSONResponse:
        return _error_response(500, str(exc))

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        messages = [Message(role=m.role, content=m.content) for m in request.messages]
        try:
            run = await completion_service.run(
                messages,
                agent=request.agent,
                model=request.model,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )
        except AgentNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "agent_not_found"}
            ) from exc
        except LLMNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "model_not_found"}
            ) from exc
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
    llm_registry, agent_registry = build_registries(config_path)
    completion_service = CompletionService(llm_registry, agent_registry)

    app = FastAPI()
    add_chat_completions_route(app, completion_service)
    return app
