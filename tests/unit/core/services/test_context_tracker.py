"""Tests for ContextFootprintTracker."""

from agent.core.services.context_tracker import ContextFootprintTracker


def test_get_given_nothing_recorded_returns_none():
    tracker = ContextFootprintTracker()

    assert tracker.get("researcher", "s1") is None


def test_get_after_record_returns_the_recorded_value():
    tracker = ContextFootprintTracker()

    tracker.record("researcher", "s1", 100)

    assert tracker.get("researcher", "s1") == 100


def test_record_twice_overwrites_rather_than_sums():
    tracker = ContextFootprintTracker()
    tracker.record("researcher", "s1", 100)

    tracker.record("researcher", "s1", 50)

    assert tracker.get("researcher", "s1") == 50


def test_record_keeps_sessions_of_different_agents_independent():
    tracker = ContextFootprintTracker()

    tracker.record("researcher", "s1", 100)
    tracker.record("writer", "s1", 50)

    assert tracker.get("researcher", "s1") == 100
    assert tracker.get("writer", "s1") == 50


def test_forget_removes_an_entry():
    tracker = ContextFootprintTracker()
    tracker.record("researcher", "s1", 100)

    tracker.forget("researcher", "s1")

    assert tracker.get("researcher", "s1") is None


def test_forget_given_unknown_session_does_not_raise():
    tracker = ContextFootprintTracker()

    tracker.forget("researcher", "does-not-exist")  # must not raise


def test_record_evicts_least_recently_touched_session_over_max_sessions():
    tracker = ContextFootprintTracker(max_sessions=2)
    tracker.record("researcher", "s1", 1)
    tracker.record("researcher", "s2", 1)

    tracker.record("researcher", "s3", 1)

    assert tracker.get("researcher", "s1") is None
    assert tracker.get("researcher", "s2") is not None
    assert tracker.get("researcher", "s3") is not None


def test_record_touching_a_session_protects_it_from_eviction():
    tracker = ContextFootprintTracker(max_sessions=2)
    tracker.record("researcher", "s1", 1)
    tracker.record("researcher", "s2", 1)

    tracker.record("researcher", "s1", 1)
    tracker.record("researcher", "s3", 1)

    assert tracker.get("researcher", "s1") is not None
    assert tracker.get("researcher", "s2") is None
    assert tracker.get("researcher", "s3") is not None


def test_record_given_max_sessions_none_never_evicts():
    tracker = ContextFootprintTracker(max_sessions=None)

    for i in range(50):
        tracker.record("researcher", f"s{i}", 1)

    assert tracker.get("researcher", "s0") is not None
