"""Tests for GetCurrentTimeTool and its parameters model."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from agent.core.tools.get_current_time import GetCurrentTimeParams, GetCurrentTimeTool


async def test_execute_returns_parseable_iso_8601_timestamp():
    tool = GetCurrentTimeTool()

    result = await tool.execute()

    datetime.fromisoformat(result)


async def test_execute_given_no_offset_defaults_to_utc():
    tool = GetCurrentTimeTool()

    result = await tool.execute()

    assert datetime.fromisoformat(result).utcoffset() == timedelta(0)


async def test_execute_given_offset_uses_it():
    tool = GetCurrentTimeTool()

    result = await tool.execute(utc_offset_minutes=330)

    assert datetime.fromisoformat(result).utcoffset() == timedelta(minutes=330)


async def test_execute_given_negative_offset_uses_it():
    tool = GetCurrentTimeTool()

    result = await tool.execute(utc_offset_minutes=-480)

    assert datetime.fromisoformat(result).utcoffset() == timedelta(minutes=-480)


def test_tool_declares_name_and_description():
    tool = GetCurrentTimeTool()

    assert tool.name == "get_current_time"
    assert tool.description


def test_tool_parameters_model_is_get_current_time_params():
    tool = GetCurrentTimeTool()

    assert tool.parameters_model is GetCurrentTimeParams


def test_params_schema_has_no_required_fields():
    schema = GetCurrentTimeParams.model_json_schema()

    assert schema.get("required", []) == []


def test_params_given_valid_offset_is_accepted():
    params = GetCurrentTimeParams(utc_offset_minutes=330)

    assert params.utc_offset_minutes == 330


def test_params_given_no_offset_defaults_to_none():
    params = GetCurrentTimeParams()

    assert params.utc_offset_minutes is None


def test_params_given_offset_too_large_raises_validation_error():
    with pytest.raises(ValidationError):
        GetCurrentTimeParams(utc_offset_minutes=1440)


def test_params_given_offset_too_small_raises_validation_error():
    with pytest.raises(ValidationError):
        GetCurrentTimeParams(utc_offset_minutes=-1440)


def test_params_given_extra_field_raises_validation_error():
    with pytest.raises(ValidationError):
        GetCurrentTimeParams.model_validate({"utc_offset_minutes": None, "unexpected": "x"})
