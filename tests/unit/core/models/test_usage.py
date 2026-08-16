from agent.core.models.usage import Usage


def test_usage_given_token_counts_constructs():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
