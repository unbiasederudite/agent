import pytest
from pydantic import ValidationError

from agent.core.models.message import Message


def test_message_given_valid_role_and_content_constructs():
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"


def test_message_given_invalid_role_raises_validation_error():
    with pytest.raises(ValidationError):
        Message(role="bogus", content="hello")
