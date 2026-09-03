"""Tests for api/logging_setup.py's JsonFormatter and RunContextFilter."""

import json
import logging
import sys

from agent.api.logging_setup import JsonFormatter, RunContextFilter
from agent.core.run_context import run_context


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="something failed: %s",
        args=("detail",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_renders_valid_json_with_expected_keys():
    formatter = JsonFormatter()
    record = _record(request_id="req-1")

    parsed = json.loads(formatter.format(record))

    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "something failed: detail"
    assert parsed["request_id"] == "req-1"
    assert "timestamp" in parsed


def test_json_formatter_given_no_request_id_omits_it():
    formatter = JsonFormatter()
    record = _record(request_id=None)

    parsed = json.loads(formatter.format(record))

    assert "request_id" not in parsed


def test_json_formatter_promotes_extra_fields_to_top_level_keys():
    formatter = JsonFormatter()
    record = _record(request_id="req-1", exception_type="RateLimitError", status_code=429)

    parsed = json.loads(formatter.format(record))

    assert parsed["exception_type"] == "RateLimitError"
    assert parsed["status_code"] == 429


def test_json_formatter_given_exc_info_includes_traceback():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record(request_id="req-1")
        record.exc_info = sys.exc_info()

    parsed = json.loads(formatter.format(record))

    assert "ValueError" in parsed["exc_info"]


def test_json_formatter_given_extra_field_matching_output_key_does_not_clobber_it():
    formatter = JsonFormatter()
    record = _record(
        request_id="req-1",
        level="CUSTOM_LEVEL",
        logger="CUSTOM_LOGGER",
        timestamp="CUSTOM_TS",
    )

    parsed = json.loads(formatter.format(record))

    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test.logger"
    assert parsed["timestamp"] != "CUSTOM_TS"


def test_json_formatter_given_agent_and_session_id_includes_them():
    formatter = JsonFormatter()
    record = _record(request_id=None, agent="researcher", session_id="sess-1")

    parsed = json.loads(formatter.format(record))

    assert parsed["agent"] == "researcher"
    assert parsed["session_id"] == "sess-1"


def test_json_formatter_given_no_agent_or_session_id_omits_them():
    formatter = JsonFormatter()
    record = _record(request_id=None, agent=None, session_id=None)

    parsed = json.loads(formatter.format(record))

    assert "agent" not in parsed
    assert "session_id" not in parsed


def test_run_context_filter_given_active_run_stamps_agent_and_session_id():
    record = _record()
    with run_context("researcher", "sess-1"):
        RunContextFilter().filter(record)

    assert record.agent == "researcher"  # type: ignore[attr-defined]
    assert record.session_id == "sess-1"  # type: ignore[attr-defined]


def test_run_context_filter_given_no_active_run_stamps_none():
    record = _record()

    RunContextFilter().filter(record)

    assert record.agent is None  # type: ignore[attr-defined]
    assert record.session_id is None  # type: ignore[attr-defined]
