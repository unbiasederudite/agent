"""FastAPI application for the backend-native agent API."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agent.api.schemas import (
    AgentRunRequest,
    AgentRunResponse,
    AgentSummary,
    ToolSummary,
)
from agent.core.exceptions import (
    AgentError,
    AgentNotFoundError,
    LLMError,
    LLMNotFoundError,
    LLMRateLimitedError,
    LLMTimeoutError,
    StrategyNotFoundError,
    ToolNotFoundError,
)
from agent.core.factories.app import build_registries
from agent.core.registries.agent import AgentRegistry
from agent.core.registries.llm import LLMRegistry
from agent.core.registries.strategy import StrategyRegistry
from agent.core.registries.tool import ToolRegistry
from agent.core.services.agent_run import AgentRunService


def add_exception_handlers(app: FastAPI) -> None:
    """Register a 400 handler for request-validation failures and a 500 catch-all on `app`.

    Every other error status (404/429/502/504/500 from `AgentError` subclasses, and
    framework-level 404/405s) is left to FastAPI's own `HTTPException`/Starlette handling,
    which already serializes `HTTPException(detail=...)` as `{"detail": ...}`.
    """

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0]
        loc = [str(p) for p in first["loc"] if p != "body"]
        message = first["msg"].removeprefix("Value error, ")
        return JSONResponse(
            status_code=400,
            content={"detail": {"message": message, "param": ".".join(loc) if loc else None}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": {"message": str(exc)}})


def add_agent_run_route(app: FastAPI, agent_run_service: AgentRunService) -> None:
    """Register POST /v1/agents/{agent_name} on `app`, backed by `agent_run_service`."""

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
            )
        except AgentNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "agent_not_found"}
            ) from exc
        except LLMNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "model_not_found"}
            ) from exc
        except StrategyNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "strategy_not_found"}
            ) from exc
        except ToolNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail={"message": str(exc), "code": "tool_not_found"}
            ) from exc
        except LLMRateLimitedError as exc:
            raise HTTPException(status_code=429, detail={"message": str(exc)}) from exc
        except LLMTimeoutError as exc:
            raise HTTPException(status_code=504, detail={"message": str(exc)}) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
        except AgentError as exc:
            raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc

        return AgentRunResponse(
            model=run.model, message=run.response, usage=run.usage, finish_reason=run.finish_reason
        )


def add_registry_routes(
    app: FastAPI,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    llm_registry: LLMRegistry,
    strategy_registry: StrategyRegistry,
) -> None:
    """Register GET /v1/agents, GET /v1/tools, GET /v1/models, GET /v1/strategies on `app`."""

    @app.get("/v1/agents")
    async def list_agents() -> list[AgentSummary]:
        return [
            AgentSummary(
                name=name,
                model=config.model,
                strategy=config.strategy,
                tools=config.tools,
            )
            for name, config in agent_registry.all().items()
        ]

    @app.get("/v1/tools")
    async def list_tools() -> list[ToolSummary]:
        return [
            ToolSummary(name=name, description=tool.description, parameters=tool.parameters)
            for name, tool in tool_registry.all().items()
        ]

    @app.get("/v1/models")
    async def list_models() -> list[str]:
        return list(llm_registry.all().keys())

    @app.get("/v1/strategies")
    async def list_strategies() -> list[str]:
        return list(strategy_registry.all().keys())


def create_app(config_path: Path) -> FastAPI:
    """Build the FastAPI app, wired from the AppConfig JSON at `config_path`."""
    llm_registry, agent_registry, tool_registry, strategy_registry, base_prompt = build_registries(
        config_path
    )
    agent_run_service = AgentRunService(
        llm_registry, agent_registry, tool_registry, strategy_registry, base_prompt
    )

    app = FastAPI()
    add_exception_handlers(app)
    add_agent_run_route(app, agent_run_service)
    add_registry_routes(app, agent_registry, tool_registry, llm_registry, strategy_registry)
    return app
