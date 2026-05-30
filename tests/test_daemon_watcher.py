"""
Tests for watcher.py - SessionWatcher poll loop, callback firing, and dedup.
"""
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frontend_adapters import ContinueAdapter
from watcher import SessionWatcher


def _write_session(path: Path, content: list) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def _make_adapter(sessions_dir: Path) -> ContinueAdapter:
    return ContinueAdapter(sessions_dir=str(sessions_dir))


def _cfg(poll_interval: float = 0.05) -> dict:
    return {
        "poll_interval": poll_interval,
        "sessions_dir": "",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for(condition_fn, timeout=2.0, interval=0.05) -> bool:
    """Poll until condition_fn() returns True or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition_fn():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Basic callback firing
# ---------------------------------------------------------------------------

class TestWatcherCallbackFiring:
    def test_changed_file_triggers_callback(self, tmp_path):
        session_path = tmp_path / "abc.json"
        _write_session(session_path, [])

        fired: list[str] = []

        def callback(session_id, adapter):
            fired.append(session_id)

        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, on_session_changed=callback, cfg=_cfg())
        watcher.start()
        try:
            # Wait for initial poll to register the file.
            time.sleep(0.15)
            # Touch the file.
            _write_session(session_path, [{"touched": True}])
            assert _wait_for(lambda: "abc" in fired, timeout=2.0)
        finally:
            watcher.stop()

    def test_unchanged_file_does_not_fire(self, tmp_path):
        session_path = tmp_path / "quiet.json"
        _write_session(session_path, [])

        fired: list[str] = []

        def callback(session_id, adapter):
            fired.append(session_id)

        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, on_session_changed=callback, cfg=_cfg())
        watcher.start()
        try:
            time.sleep(0.3)
            # File not touched since initial registration.
            # Only the initial "discovery" pass fires (mtime starts at 0.0).
            initial_count = len([f for f in fired if f == "quiet"])
            time.sleep(0.3)
            # Should not fire again.
            assert len([f for f in fired if f == "quiet"]) == initial_count
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# New file discovery
# ---------------------------------------------------------------------------

class TestNewFileDiscovery:
    def test_discovers_new_session_after_start(self, tmp_path):
        fired: list[str] = []

        def callback(session_id, adapter):
            fired.append(session_id)

        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, on_session_changed=callback, cfg=_cfg())
        watcher.start()
        try:
            # No files yet; wait for at least one poll.
            time.sleep(0.15)
            # Create a new session file.
            _write_session(tmp_path / "brand_new.json", [])
            assert _wait_for(lambda: "brand_new" in fired, timeout=2.0)
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# sessions.json excluded
# ---------------------------------------------------------------------------

class TestSessionsJsonExcluded:
    def test_sessions_json_never_fires(self, tmp_path):
        (tmp_path / "sessions.json").write_text("{}", encoding="utf-8")

        fired: list[str] = []

        def callback(session_id, adapter):
            fired.append(session_id)

        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, on_session_changed=callback, cfg=_cfg())
        watcher.start()
        try:
            time.sleep(0.3)
            assert "sessions" not in fired
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# Dedup: no double-fire while in_flight
# ---------------------------------------------------------------------------

class TestInFlightDedup:
    def test_queued_but_not_double_fired(self, tmp_path):
        """
        If a callback is slow and the file changes again while it runs,
        the second change is queued and fired exactly once after the first
        finishes - not fired twice concurrently.
        """
        session_path = tmp_path / "slow.json"
        _write_session(session_path, [])

        call_count = [0]
        max_concurrent = [0]
        active = [0]
        lock = threading.Lock()

        def slow_callback(session_id, adapter):
            with lock:
                active[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active[0])
            time.sleep(0.1)
            call_count[0] += 1
            with lock:
                active[0] -= 1

        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, on_session_changed=slow_callback, cfg=_cfg(0.02))
        watcher.start()
        try:
            time.sleep(0.15)  # let initial discovery settle
            # Touch the file twice in quick succession.
            _write_session(session_path, [{"v": 1}])
            time.sleep(0.01)
            _write_session(session_path, [{"v": 2}])
            # Wait for both cycles to complete.
            _wait_for(lambda: call_count[0] >= 2, timeout=3.0)
            # The max concurrent callbacks should be 1 (serial executor).
            assert max_concurrent[0] == 1
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

class TestWatcherStatus:
    def test_status_reports_running(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, cfg=_cfg())
        watcher.start()
        try:
            status = watcher.status()
            assert status["running"] is True
        finally:
            watcher.stop()

    def test_status_shows_watched_sessions(self, tmp_path):
        _write_session(tmp_path / "one.json", [])
        adapter = _make_adapter(tmp_path)
        watcher = SessionWatcher(adapter=adapter, cfg=_cfg())
        watcher.start()
        try:
            _wait_for(lambda: len(watcher.status()["watching"]) > 0, timeout=2.0)
            assert "one" in watcher.status()["watching"]
        finally:
            watcher.stop()


