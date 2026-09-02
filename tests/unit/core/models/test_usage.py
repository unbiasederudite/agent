import pytest

from agent.core.models.usage import Usage, sum_usage


def test_usage_given_token_counts_constructs():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_usage_given_no_cost_defaults_to_none():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert usage.cost_usd is None


def test_usage_given_cost_constructs():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.0023)

    assert usage.cost_usd == 0.0023


def test_sum_usage_adds_token_counts():
    a = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)

    result = sum_usage(a, b)

    assert result.prompt_tokens == 13
    assert result.completion_tokens == 7
    assert result.total_tokens == 20


def test_sum_usage_given_both_costs_known_adds_them():
    a = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.01)
    b = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.02)

    assert sum_usage(a, b).cost_usd == pytest.approx(0.03)


def test_sum_usage_given_one_cost_unknown_treats_it_as_zero():
    a = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.01)
    b = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=None)

    assert sum_usage(a, b).cost_usd == pytest.approx(0.01)


def test_sum_usage_given_both_costs_unknown_stays_none():
    a = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=None)
    b = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=None)

    assert sum_usage(a, b).cost_usd is None
