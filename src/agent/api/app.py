"""FastAPI application for the backend-native agent API."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.api.logging_setup import RunContextFilter, configure_logging
from agent.api.request_context import (
    INTERNAL_ERROR_MESSAGE,
    RequestIdFilter,
    RequestIdMiddleware,
    current_request_id,
)
from agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentSummary,
    AgentUsageResponse,
    SessionHistoryResponse,
    SessionUsageResponse,
    ToolSummary,
)
from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    CompactionExhaustedError,
    ConfigError,
    InputTooLargeError,
    LLMContextWindowExceededError,
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
from agent.core.factories.app import build_registries
from agent.core.models.config import AppConfig
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.services.agent_run import AgentRunService
from agent.core.services.compaction import CompactionService
from agent.core.services.context_tracker import ContextFootprintTracker
from agent.core.services.cost_tracker import CostTracker
from agent.core.services.session_service import SessionService
from agent.core.session_stores.in_memory import InMemorySessionStore

logger = logging.getLogger(__name__)

_RETRY_AFTER_SECONDS = "5"

_UNIFORM_ERROR_MAP: dict[type[Exception], tuple[int, str]] = {
    AgentNotFoundError: (404, "agent_not_found"),
    LLMNotFoundError: (404, "model_not_found"),
    StrategyNotFoundError: (404, "strategy_not_found"),
    SessionNotFoundError: (404, "session_not_found"),
    ToolNotFoundError: (404, "tool_not_found"),
    InputTooLargeError: (413, "input_too_large"),
    SessionBusyError: (409, "session_busy"),
    ToolNotAllowedError: (403, "tool_not_allowed"),
    ModelNotAllowedError: (403, "model_not_allowed"),
    StrategyNotAllowedError: (403, "strategy_not_allowed"),
}


def add_exception_handlers(app: FastAPI) -> None:
    """Register the validation-error, HTTPException, and 500 catch-all handlers on `app`.

    Args:
        app: FastAPI app to register the route on.
    """

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Map the first validation error to a 400 with a normalized `message`/`param`.

        Args:
            _request: The failed request (unused).
            exc: The validation error.

        Returns:
            JSONResponse: 400 with the normalized error body.
        """
        first = exc.errors()[0]
        loc = [str(p) for p in first["loc"] if p != "body"]
        message = first["msg"].removeprefix("Value error, ")
        logger.info("request validation failed: %s", message)
        return JSONResponse(
            status_code=400,
            content={"detail": {"message": message, "param": ".".join(loc) if loc else None}},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Normalize an `HTTPException` into a JSON response, logging 404s and 405s.

        Args:
            request: The failed request.
            exc: The raised HTTPException.

        Returns:
            JSONResponse: response mirroring `exc`'s status code, detail, and headers.
        """
        if exc.status_code in (404, 405):
            logger.info("%d response: %s %s", exc.status_code, request.method, request.url.path)
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail}
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_request: Request, exc: Exception) -> JSONResponse:
        """Map any otherwise-unhandled exception to a 500 with the current request id.

        Args:
            _request: The failed request (unused).
            exc: The unhandled exception.

        Returns:
            JSONResponse: 500 with a fixed message and the current request id.
        """
        logger.error("unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": {"message": INTERNAL_ERROR_MESSAGE, "request_id": current_request_id()}
            },
        )


def add_agent_run_route(app: FastAPI, agent_run_service: AgentRunService) -> None:
    """Register POST /v1/agents/{agent_name} on `app`, backed by `agent_run_service`.

    Args:
        app: FastAPI app to register the route on.
        agent_run_service: Service backing the route.
    """

    @app.post("/v1/agents/{agent_name}")
    async def run_agent(agent_name: str, request: AgentRunRequest) -> AgentRunResponse:
        try:
            run = await agent_run_service.run(
                request.message,
                agent_name,
                model=request.model,
                strategy=request.strategy,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                tools=request.tools,
                session_id=request.session_id,
            )
        except tuple(_UNIFORM_ERROR_MAP) as exc:
            status_code, code = _UNIFORM_ERROR_MAP[type(exc)]
            raise HTTPException(
                status_code=status_code, detail={"message": str(exc), "code": code}
            ) from exc
        except LLMRateLimitedError as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "The model provider is rate-limiting requests. "
                    "Please try again shortly."
                },
                headers={"Retry-After": _RETRY_AFTER_SECONDS},
            ) from exc
        except LLMTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "The request to the model provider timed out. Please try again."
                },
            ) from exc
        except CompactionExhaustedError as exc:
            raise HTTPException(
                status_code=413,
                detail={
                    "message": "This conversation is too large for the model's context window "
                    "even after summarization. Start a new session or switch to a model with a "
                    "larger context window.",
                    "code": "compaction_exhausted",
                },
            ) from exc
        except LLMContextWindowExceededError as exc:
            raise HTTPException(
                status_code=413,
                detail={
                    "message": "This request is too large for the model's context window and "
                    "could not be reduced automatically.",
                    "code": "context_window_exceeded",
                },
            ) from exc
        except LLMOverloadedError as exc:
            raise HTTPException(
                status_code=503,
                detail={"message": "This model is at capacity. Please try again shortly."},
                headers={"Retry-After": _RETRY_AFTER_SECONDS},
            ) from exc
        except RequestTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail={"message": "This request exceeded its configured time budget."},
            ) from exc
        except LLMError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "The request to the model provider failed. Please try again.",
                    "request_id": current_request_id(),
                },
            ) from exc
        except AgentError as exc:
            logger.error("unhandled AgentError subtype: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"message": INTERNAL_ERROR_MESSAGE, "request_id": current_request_id()},
            ) from exc

        return AgentRunResponse(
            model=run.model,
            message=run.response,
            usage=run.usage,
            finish_reason=run.finish_reason,
            session_id=run.session_id,
        )


def add_registry_routes(
    app: FastAPI,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    llm_registry: LLMRegistry,
    strategy_registry: StrategyRegistry,
) -> None:
    """Register GET /v1/agents, GET /v1/tools, GET /v1/models, GET /v1/strategies on `app`.

    Args:
        app: FastAPI app to register the routes on.
        agent_registry: Registry backing GET /v1/agents.
        tool_registry: Registry backing GET /v1/tools.
        llm_registry: Registry backing GET /v1/models.
        strategy_registry: Registry backing GET /v1/strategies.
    """

    @app.get("/v1/agents")
    async def list_agents() -> list[AgentSummary]:
        agents = agent_registry.all()
        logger.debug("listed agents: %d entries", len(agents))
        return [
            AgentSummary(
                name=name,
                model=config.model,
                strategy=config.strategy,
                tools=config.tools,
            )
            for name, config in agents.items()
        ]

    @app.get("/v1/tools")
    async def list_tools() -> list[ToolSummary]:
        tools = tool_registry.all()
        logger.debug("listed tools: %d entries", len(tools))
        return [
            ToolSummary(
                name=name,
                description=tool.description,
                parameters=tool.parameters_model.model_json_schema(),
            )
            for name, tool in tools.items()
        ]

    @app.get("/v1/models")
    async def list_models() -> list[str]:
        models = list(llm_registry.all().keys())
        logger.debug("listed models: %d entries", len(models))
        return models

    @app.get("/v1/strategies")
    async def list_strategies() -> list[str]:
        strategies = list(strategy_registry.all().keys())
        logger.debug("listed strategies: %d entries", len(strategies))
        return strategies


def add_health_route(app: FastAPI) -> None:
    """Register a dependency-free `GET /health` liveness route on `app`.

    Args:
        app: FastAPI app to register the route on.
    """

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}


def add_session_routes(app: FastAPI, session_service: SessionService) -> None:
    """Register GET/DELETE /v1/agents/{agent_name}/sessions/{session_id} and its /usage on `app`.

    Args:
        app: FastAPI app to register the routes on.
        session_service: Service backing the routes.
    """

    @app.get("/v1/agents/{agent_name}/sessions/{session_id}")
    async def get_session(agent_name: str, session_id: str) -> SessionHistoryResponse:
        try:
            messages = await session_service.get_history(agent_name, session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "session_not_found"}
            ) from exc
        return SessionHistoryResponse(session_id=session_id, messages=messages)

    @app.get("/v1/agents/{agent_name}/sessions/{session_id}/usage")
    async def get_session_usage(agent_name: str, session_id: str) -> SessionUsageResponse:
        try:
            cumulative, context_tokens = await session_service.get_usage(agent_name, session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "session_not_found"}
            ) from exc
        return SessionUsageResponse(
            session_id=session_id,
            cumulative=cumulative,
            context_tokens=context_tokens,
        )

    @app.delete("/v1/agents/{agent_name}/sessions/{session_id}", status_code=204)
    async def delete_session(agent_name: str, session_id: str) -> None:
        try:
            await session_service.delete(agent_name, session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "session_not_found"}
            ) from exc
        except SessionBusyError as exc:
            raise HTTPException(
                status_code=409, detail={"message": str(exc), "code": "session_busy"}
            ) from exc


def add_usage_routes(
    app: FastAPI, agent_registry: AgentRegistry, cost_tracker: CostTracker
) -> None:
    """Register GET /v1/agents/{agent_name}/usage on `app`.

    Args:
        app: FastAPI app to register the route on.
        agent_registry: Registry used to validate the agent name exists.
        cost_tracker: Tracker backing the route.
    """

    @app.get("/v1/agents/{agent_name}/usage")
    async def get_agent_usage(agent_name: str) -> AgentUsageResponse:
        try:
            agent_registry.get(agent_name)
        except AgentNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "agent_not_found"}
            ) from exc
        return AgentUsageResponse(agent=agent_name, cumulative=cost_tracker.agent_usage(agent_name))


def create_app(config_path: Path) -> FastAPI:
    """Build the FastAPI app, wired from the startup configuration JSON at `config_path`.

    Args:
        config_path: Path to the startup configuration JSON file.

    Returns:
        FastAPI: the built app.

    Raises:
        ConfigError: if `config_path` is missing, not valid JSON, or fails validation.
    """
    try:
        config = AppConfig.model_validate_json(config_path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ConfigError(str(exc)) from exc

    (
        llm_registry,
        agent_registry,
        tool_registry,
        strategy_registry,
        base_prompt,
        compaction_config,
        logging_config,
        max_sessions,
    ) = build_registries(config)
    configure_logging(logging_config, RequestIdFilter(), RunContextFilter())
    session_store = InMemorySessionStore(max_sessions=max_sessions)
    cost_tracker = CostTracker(max_sessions=max_sessions)
    context_tracker = ContextFootprintTracker(max_sessions=max_sessions)
    compaction_service = (
        CompactionService(llm_registry, session_store, compaction_config, context_tracker)
        if compaction_config is not None
        else None
    )
    agent_run_service = AgentRunService(
        llm_registry,
        agent_registry,
        tool_registry,
        strategy_registry,
        base_prompt,
        session_store,
        compaction_service,
        cost_tracker=cost_tracker,
        context_tracker=context_tracker,
    )
    session_service = SessionService(
        session_store, cost_tracker=cost_tracker, context_tracker=context_tracker
    )

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    add_exception_handlers(app)
    add_agent_run_route(app, agent_run_service)
    add_registry_routes(app, agent_registry, tool_registry, llm_registry, strategy_registry)
    add_health_route(app)
    add_session_routes(app, session_service)
    add_usage_routes(app, agent_registry, cost_tracker)
    logger.info(
        "agent-core started: %d agent(s), %d tool(s), %d LLM(s), %d strategy(s), compaction=%s",
        len(agent_registry.all()),
        len(tool_registry.all()),
        len(llm_registry.all()),
        len(strategy_registry.all()),
        "enabled" if compaction_config is not None else "disabled",
    )
    return app
