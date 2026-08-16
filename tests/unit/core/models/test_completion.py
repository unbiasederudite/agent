from agent.core.models.completion import Completion
from agent.core.models.message import Message
from agent.core.models.usage import Usage


def test_completion_given_message_and_usage_constructs():
    completion = Completion(
        message=Message(role="assistant", content="hi"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )

    assert completion.message.content == "hi"
    assert completion.usage.total_tokens == 2
    assert completion.finish_reason == "stop"
