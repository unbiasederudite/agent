"""Tests for core/run_context.py -- the (agent, session_id) correlation ContextVar."""

from agent.core.run_context import current_run_context, run_context, update_session_id


def test_current_run_context_given_no_active_run_returns_none():
    assert current_run_context() is None


def test_run_context_given_active_block_returns_agent_and_session_id():
    with run_context("researcher", "sess-1"):
        assert current_run_context() == ("researcher", "sess-1")


def test_run_context_given_no_session_id_returns_agent_and_none():
    with run_context("researcher", None):
        assert current_run_context() == ("researcher", None)


def test_run_context_given_block_exits_restores_none():
    with run_context("researcher", "sess-1"):
        pass

    assert current_run_context() is None


def test_run_context_given_nested_blocks_restores_outer_on_exit():
    with run_context("researcher", "sess-1"):
        with run_context("writer", "sess-2"):
            assert current_run_context() == ("writer", "sess-2")
        assert current_run_context() == ("researcher", "sess-1")


def test_update_session_id_given_active_run_updates_session_id():
    with run_context("researcher", None):
        update_session_id("sess-new")

        assert current_run_context() == ("researcher", "sess-new")


def test_update_session_id_given_no_active_run_is_a_no_op():
    update_session_id("sess-new")

    assert current_run_context() is None
