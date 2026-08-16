from agent.core.models.message import Message
from agent.core.models.run import Run
from agent.core.models.usage import Usage


def test_run_given_fields_constructs():
    run = Run(
        model="openai/gpt-4o",
        request=[Message(role="user", content="hi")],
        response=Message(role="assistant", content="hello"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )

    assert run.model == "openai/gpt-4o"
    assert run.request[0].content == "hi"
    assert run.response.content == "hello"
    assert run.usage.total_tokens == 2
    assert run.finish_reason == "stop"
