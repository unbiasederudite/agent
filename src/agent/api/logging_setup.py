"""Logging setup: format (text/json), console/file handlers, rotation, filters.

`configure_logging()` performs I/O (opening files, wiring handlers), which `core/` (per
its own README) owns none of -- it lives in `api/`, called once from `create_app()`.
`JsonFormatter`/`RunContextFilter` don't do I/O themselves, but they're process-level
logging infrastructure, not agent intelligence, so they live here alongside the function
that wires them up rather than in `core/`.
"""

import json
import logging
import logging.handlers

from agent.core.exceptions import ConfigError
from agent.core.models.config import LoggingConfig
from agent.core.run_context import current_run_context

# Derived from a real LogRecord rather than hand-listed, so it can't drift out of sync
# with whatever attributes this Python version actually sets by default (e.g. `taskName`,
# added in 3.12) -- a missed one would otherwise silently leak into JsonFormatter's
# "extra fields" output as noise.
_LOGRECORD_DEFAULT_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys())

# Keys JsonFormatter.format() writes into `payload` itself, above the "everything else is
# an extra field" loop -- names this project invented (or, for message/asctime, names a
# *different* formatter might set upstream), so Python has no built-in list of these the
# way _LOGRECORD_DEFAULT_ATTRS is derived; must stay hand-written.
_FORMATTER_OWN_KEYS = frozenset(
    {"message", "asctime", "timestamp", "level", "logger", "request_id", "agent", "session_id"}
)

_STANDARD_ATTRS = _LOGRECORD_DEFAULT_ATTRS | _FORMATTER_OWN_KEYS


class JsonFormatter(logging.Formatter):
    """Renders each `LogRecord` as one JSON object per line.

    Includes any `extra=` fields a call site attached (e.g. `exception_type`, `status_code`)
    as their own top-level keys, not just baked into the formatted `message` string -- that
    distinction is what makes "how many retries were rate-limits vs. 5xxs" an actual
    queryable question in a log aggregator, not just readable text.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON-serialized string containing the log record's data.
        """
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id
        agent = getattr(record, "agent", None)
        if agent is not None:
            payload["agent"] = agent
        session_id = getattr(record, "session_id", None)
        if session_id is not None:
            payload["session_id"] = session_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class RunContextFilter(logging.Filter):
    """Stamps `record.agent`/`record.session_id` from the current run's `ContextVar`.

    Mirrors `RequestIdFilter` exactly, but for `core.run_context` instead of the
    request id -- attached to a log *handler*, so it applies uniformly regardless of
    which module's `logging.getLogger(...)` emitted the record. Both are `None` outside
    an `AgentRunService.run()` call.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the current run's agent/session_id to the log record, or None if outside one."""
        context = current_run_context()
        record.agent, record.session_id = context if context is not None else (None, None)
        return True


def configure_logging(config: LoggingConfig, *extra_filters: logging.Filter) -> None:
    """Wire up console/file handlers, the chosen formatter, and the level from `config`.

    Replaces `core/factories/app.py`'s old bare `logging.basicConfig(level=...)` call.
    `extra_filters` (`RequestIdFilter`, `RunContextFilter`) are attached to every handler
    this creates, so every log line -- from `core/`, `adapters/`, or `api/` -- carries the
    request id of whichever request (if any), and the agent/session_id of whichever run
    (if any), it was logged during.

    Raises:
        ConfigError: if `config.file` is set but can't be opened (bad directory, no write
            permission) -- reported through the same fatal-startup path as any other
            config mistake, not a raw `OSError`.
    """
    formatter: logging.Formatter = (
        JsonFormatter()
        if config.format == "json"
        else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "[request_id=%(request_id)s agent=%(agent)s session=%(session_id)s] %(message)s",
            # Defaults guard against a KeyError if a handler is ever configured without
            # RequestIdFilter/RunContextFilter attached -- both filters always supply
            # these in practice, but the format string shouldn't crash logging itself
            # if one is ever missing.
            defaults={"request_id": None, "agent": None, "session_id": None},
        )
    )
    handlers: list[logging.Handler] = []
    if config.console:
        handlers.append(logging.StreamHandler())
    if config.file is not None:
        try:
            file_handler: logging.Handler = (
                logging.FileHandler(config.file)
                if config.file_max_bytes is None
                else logging.handlers.RotatingFileHandler(
                    config.file,
                    maxBytes=config.file_max_bytes,
                    backupCount=config.file_backup_count,
                )
            )
        except OSError as exc:
            raise ConfigError(f"cannot open log file '{config.file}': {exc}") from exc
        handlers.append(file_handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        for extra_filter in extra_filters:
            handler.addFilter(extra_filter)
    logging.basicConfig(level=config.level, handlers=handlers, force=True)
