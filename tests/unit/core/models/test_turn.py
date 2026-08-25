"""Tests for Turn -- the aggregate result of one IStrategy.run() call."""

import pytest
from pydantic import ValidationError

from agent.core.models.message import Message, ToolCall, ToolCallFunction
from agent.core.models.turn import Turn
from agent.core.models.usage import Usage


def _usage() -> Usage:
    return Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)


def test_turn_message_given_single_message_returns_it():
    final = Message(role="assistant", content="done")

    turn = Turn(messages=[final], usage=_usage(), finish_reason="stop")

    assert turn.message == final


def test_turn_message_given_multiple_messages_returns_the_last_one():
    tool_call_message = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="echo", arguments="{}"))],
    )
    tool_result = Message(role="tool", tool_call_id="call_1", name="echo", content="hi")
    final = Message(role="assistant", content="done")

    turn = Turn(
        messages=[tool_call_message, tool_result, final], usage=_usage(), finish_reason="stop"
    )

    assert turn.message == final
    assert turn.message is turn.messages[-1]


def test_turn_given_fields_constructs():
    final = Message(role="assistant", content="done")

    turn = Turn(messages=[final], usage=_usage(), finish_reason="length")

    assert turn.messages == [final]
    assert turn.usage.total_tokens == 2
    assert turn.finish_reason == "length"


def test_turn_given_empty_messages_raises_validation_error():
    with pytest.raises(ValidationError):
        Turn(messages=[], usage=_usage(), finish_reason="stop")
