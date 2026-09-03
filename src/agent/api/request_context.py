"""Request correlation id, threaded through logging and set once per request."""

import logging
import re
import uuid
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "An unexpected error occurred."

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def current_request_id() -> str | None:
    """Return the current request's correlation id, or `None` outside a request context.

    Returns:
        str | None: the current request id, or `None`.
    """
    return _request_id.get()


class RequestIdMiddleware:
    """Assign or echo an `X-Request-ID` per request."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap `app`, the next layer in the ASGI middleware/router chain.

        Args:
            app: The next ASGI layer.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind the request id for one HTTP request; pass everything else through unchanged.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive channel.
            send: The ASGI send channel.

        Raises:
            Exception: whatever `self.app` raised, if a response was already partway out
                when it failed.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client_request_id = Headers(scope=scope).get("x-request-id")
        request_id = (
            client_request_id
            if client_request_id is not None and _SAFE_REQUEST_ID.fullmatch(client_request_id)
            else uuid.uuid4().hex
        )
        token = _request_id.set(request_id)
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.error("unhandled exception", exc_info=True)
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content={"detail": {"message": INTERNAL_ERROR_MESSAGE, "request_id": request_id}},
            )
            response.headers["X-Request-ID"] = request_id
            await response(scope, receive, send)
        finally:
            _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Stamps `record.request_id` from the current request's `ContextVar` onto every `LogRecord`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the current request_id to the log record, or None if outside a request.

        Args:
            record: The log record to stamp.

        Returns:
            bool: always `True` — never filters a record out.
        """
        record.request_id = current_request_id()
        return True
