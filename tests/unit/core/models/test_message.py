import pytest
from pydantic import ValidationError

from agent.core.models.message import Message, ToolCall, ToolCallFunction


def test_message_given_valid_role_and_content_constructs():
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"


def test_message_given_invalid_role_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="bogus", content="hello")


def test_message_given_neither_content_nor_tool_calls_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="assistant")


def test_message_given_empty_tool_calls_list_and_no_content_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="assistant", tool_calls=[])


def test_message_given_content_only_defaults_tool_calls_to_none():
    message = Message(role="assistant", content="hi")

    assert message.tool_calls is None


def test_message_given_tool_calls_constructs():
    tool_calls = [
        ToolCall(id="call_1", function=ToolCallFunction(name="get_current_time", arguments="{}"))
    ]

    message = Message(role="assistant", content=None, tool_calls=tool_calls)

    assert message.content is None
    assert message.tool_calls == tool_calls


def test_tool_call_given_fields_constructs():
    tool_call = ToolCall(
        id="call_1",
        function=ToolCallFunction(name="get_current_time", arguments='{"tz": "UTC"}'),
    )

    assert tool_call.id == "call_1"
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_current_time"
    assert tool_call.function.arguments == '{"tz": "UTC"}'


def test_message_given_tool_calls_dumps_nested_wire_shape():
    message = Message(
        role="assistant",
        tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="foo", arguments="{}"))],
    )

    assert message.model_dump(exclude_none=True) == {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
        ],
    }


def test_message_given_tool_role_without_tool_call_id_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="tool", content="42")


def test_message_given_tool_role_without_name_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="tool", tool_call_id="call_1", content="42")


def test_message_given_tool_role_without_content_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="tool", tool_call_id="call_1", name="get_current_time")


def test_message_given_tool_role_with_tool_calls_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(
            role="tool",
            tool_call_id="call_1",
            name="get_current_time",
            content="42",
            tool_calls=[ToolCall(id="call_2", function=ToolCallFunction(name="f", arguments="{}"))],
        )


def test_message_given_tool_role_with_id_name_and_content_constructs():
    message = Message(role="tool", tool_call_id="call_1", name="get_current_time", content="42")

    assert message.role == "tool"
    assert message.tool_call_id == "call_1"
    assert message.name == "get_current_time"
    assert message.content == "42"


def test_message_given_non_tool_role_defaults_tool_call_id_to_none():
    message = Message(role="user", content="hi")

    assert message.tool_call_id is None
