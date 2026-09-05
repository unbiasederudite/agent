"""Logging setup: JSON/text formatting, console/file handlers with rotation, and filters."""

import json
import logging
import logging.handlers

from agent.core.exceptions import ConfigError
from agent.core.models.config import LoggingConfig
from agent.core.run_context import current_run_context

_LOGRECORD_DEFAULT_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys())

_FORMATTER_OWN_KEYS = frozenset(
    {"message", "asctime", "timestamp", "level", "logger", "request_id", "agent", "session_id"}
)

_STANDARD_ATTRS = _LOGRECORD_DEFAULT_ATTRS | _FORMATTER_OWN_KEYS


class JsonFormatter(logging.Formatter):
    """Renders each `LogRecord` as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: the log record as a JSON line.
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
    """Stamps `record.agent`/`record.session_id` from the current run's context."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the current run's agent/session_id to the log record, or None if outside one.

        Args:
            record: The log record to stamp.

        Returns:
            bool: always `True` — never filters a record out.
        """
        context = current_run_context()
        record.agent, record.session_id = context if context is not None else (None, None)
        return True


def configure_logging(config: LoggingConfig, *extra_filters: logging.Filter) -> None:
    """Wire up console/file handlers, the chosen formatter, and the level from `config`.

    Args:
        config: Logging settings (level, format, output destinations).
        extra_filters: Additional log filters to attach.

    Raises:
        ConfigError: if `config.file` is set but can't be opened.
    """
    formatter: logging.Formatter = (
        JsonFormatter()
        if config.format == "json"
        else logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "[request_id=%(request_id)s agent=%(agent)s session=%(session_id)s] %(message)s",
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
