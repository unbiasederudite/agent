"""Tests for api/request_context.py -- the request correlation id middleware and filter."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from agent.api.request_context import RequestIdFilter, RequestIdMiddleware, current_request_id


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    return app


def test_request_id_middleware_given_no_client_header_generates_one_and_echoes_it():
    client = TestClient(_app())

    response = client.get("/ok")

    assert response.headers["X-Request-ID"]


def test_request_id_middleware_given_client_header_echoes_it_unchanged():
    client = TestClient(_app())

    response = client.get("/ok", headers={"X-Request-ID": "client-supplied-id"})

    assert response.headers["X-Request-ID"] == "client-supplied-id"


def test_request_id_middleware_given_two_requests_generates_different_ids():
    client = TestClient(_app())

    first = client.get("/ok").headers["X-Request-ID"]
    second = client.get("/ok").headers["X-Request-ID"]

    assert first != second


def test_request_id_middleware_given_malicious_header_generates_a_fresh_id_instead():
    client = TestClient(_app())

    response = client.get(
        "/ok", headers={"X-Request-ID": "abc] injected-agent=evil session=fake ["}
    )

    echoed = response.headers["X-Request-ID"]
    assert echoed != "abc] injected-agent=evil session=fake ["
    assert "]" not in echoed
    assert "[" not in echoed


def test_request_id_middleware_given_header_with_disallowed_characters_ignores_it():
    client = TestClient(_app())

    response = client.get("/ok", headers={"X-Request-ID": "has spaces"})

    assert response.headers["X-Request-ID"] != "has spaces"


def test_request_id_middleware_given_safe_header_characters_echoes_it_unchanged():
    client = TestClient(_app())

    response = client.get("/ok", headers={"X-Request-ID": "req-123.abc_DEF"})

    assert response.headers["X-Request-ID"] == "req-123.abc_DEF"


def test_request_id_middleware_present_on_error_response_too():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    assert response.headers["X-Request-ID"]
    request_id = response.headers["X-Request-ID"]
    assert response.json() == {
        "detail": {
            "message": "An unexpected error occurred.",
            "request_id": request_id,
        }
    }


def test_current_request_id_given_no_active_request_returns_none():
    assert current_request_id() is None


def test_request_id_middleware_sets_context_var_visible_during_the_request():
    seen: list[str | None] = []
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/check")
    async def check() -> dict[str, str]:
        seen.append(current_request_id())
        return {"status": "ok"}

    client = TestClient(app)
    client.get("/check", headers={"X-Request-ID": "abc123"})

    assert seen == ["abc123"]


def test_request_id_filter_given_no_active_request_stamps_none():
    filter_ = RequestIdFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hi", args=(), exc_info=None
    )

    filter_.filter(record)

    assert record.request_id is None


def test_request_id_filter_stamps_record_with_the_active_request_id():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture()
    handler.addFilter(RequestIdFilter())
    test_logger = logging.getLogger("agent.api.request_context.test")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    @app.get("/log")
    async def log_something() -> dict[str, str]:
        test_logger.info("inside a request")
        return {"status": "ok"}

    client = TestClient(app)
    client.get("/log", headers={"X-Request-ID": "req-xyz"})
    test_logger.removeHandler(handler)

    [record] = captured
    assert record.request_id == "req-xyz"


def test_request_id_middleware_given_unhandled_exception_logs_error(caplog):
    caplog.set_level(logging.ERROR, logger="agent.api.request_context")
    client = TestClient(_app(), raise_server_exceptions=False)

    client.get("/boom")

    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None


async def test_request_id_middleware_given_non_http_scope_passes_through_unchanged():
    calls: list[Scope] = []

    async def _stub_app(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(scope)

    middleware = RequestIdMiddleware(_stub_app)
    scope: Scope = {"type": "lifespan"}

    async def _receive() -> Message:
        raise AssertionError("should not be called")

    async def _send(message: Message) -> None:
        raise AssertionError("should not be called")

    await middleware(scope, _receive, _send)

    assert calls == [scope]
    assert current_request_id() is None


async def test_request_id_middleware_given_exception_after_response_started_reraises():
    async def _streaming_then_broken_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("boom mid-stream")

    middleware = RequestIdMiddleware(_streaming_then_broken_app)
    scope: Scope = {"type": "http", "headers": []}
    sent: list[Message] = []

    async def _receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: Message) -> None:
        sent.append(message)

    with pytest.raises(RuntimeError, match="boom mid-stream"):
        await middleware(scope, _receive, _send)

    # The real response.start already went out; no second one was attempted.
    assert len(sent) == 1
