"""Request correlation id: a ContextVar threaded through logging via a Filter.

Set by one piece of middleware per request. Lives entirely in `api/` -- `core/` and
`adapters/` modules log normally with zero awareness of request ids; the id is stamped at
the log handler level regardless of which logger emitted the record.

`RequestIdMiddleware` is pure ASGI (implements `__call__(scope, receive, send)` directly),
not the `@app.middleware("http")`/`BaseHTTPMiddleware` style -- the latter runs the
downstream app in a separate task, which Starlette's own maintainers document as not
reliably propagating `ContextVar` changes in every case (into `BackgroundTasks`, and a
closed-as-not-planned Starlette bug where a var can leak between requests under multipart
form data with a reused connection). Neither applies to this app today (no background
tasks, no multipart bodies anywhere), but the pure-ASGI form has no such gap at all, at
the cost of manually wrapping `send` to inject the response header.
"""

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

# A client-supplied X-Request-ID flows unescaped into text-format log lines (see
# logging_setup.py) and is echoed back in a response header -- restricting it to a plain
# token charset closes log-injection (forging fake bracket-delimited fields like
# "agent=evil") rather than trying to escape every dangerous character after the fact. A
# value that doesn't match this is treated the same as no header at all: a fresh id is
# generated instead of using the client's.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def current_request_id() -> str | None:
    """Return the current request's correlation id, or `None` outside a request context."""
    return _request_id.get()


class RequestIdMiddleware:
    """Assign (or echo) an `X-Request-ID` for the duration of one request.

    Reads the incoming `X-Request-ID` header if the caller already supplied one, else
    generates a new one. Every log line emitted while handling this request carries it
    (via `RequestIdFilter`), and every response -- success or error alike -- echoes it back
    in the `X-Request-ID` response header.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap `app`, the next layer in the ASGI middleware/router chain."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind the request id for one HTTP request; pass everything else through unchanged."""
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
            # This except clause intercepts unhandled exceptions before ServerErrorMiddleware.
            # Once wired into the real app, this becomes the de facto exception handler for the
            # request context, making app.py's @app.exception_handler(Exception) unreachable by
            # design (Starlette's middleware layering). Match the API's JSON error shape.
            if response_started:
                # A response is already partway out (e.g. a streaming body) -- sending a
                # second one would violate the ASGI protocol. Nothing safe left to do but
                # let it propagate to ServerErrorMiddleware, same as an exception raised
                # with no middleware here at all.
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
    """Stamps `record.request_id` from the current request's `ContextVar` onto every `LogRecord`.

    Attached to a log *handler* (not a specific logger), so it applies uniformly to every
    record that reaches that handler regardless of which module's `logging.getLogger(...)`
    emitted it -- `core/` and `adapters/` modules need no awareness of this at all.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the current request_id to the log record, or None if outside a request."""
        record.request_id = current_request_id()
        return True
