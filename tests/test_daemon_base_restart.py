"""Tests for daemon_base.py restart token mechanism.

Covers:
  - _restart_token_path: returns correct path under HUBRIS_HOME/restart_tokens/
  - _check_restart_token: returns False when no token file exists
  - _check_restart_token: returns True when token file exists
  - _consume_restart_token: deletes the token file
  - _consume_restart_token: does not raise when file is already gone
  - run() exits cleanly when restart token is present at loop start
  - run() continues normally when no token is present
"""

from pathlib import Path
from unittest import mock

import config as _config
from daemon_base import DaemonBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daemon(tmp_path: Path, name: str = "test_daemon") -> DaemonBase:
    """Create a minimal DaemonBase instance with no real action handlers."""
    return DaemonBase(
        name=name,
        root=tmp_path,
        action_handlers={},
        actor="TestActor",
        poll_interval=0.0,
    )


# ---------------------------------------------------------------------------
# _restart_token_path
# ---------------------------------------------------------------------------


def test_restart_token_path_is_under_hubris_home() -> None:
    d = _make_daemon(Path("."), "myservice")
    expected = _config.HUBRIS_HOME / "restart_tokens" / "myservice.restart"
    assert d._restart_token_path == expected


# ---------------------------------------------------------------------------
# _check_restart_token
# ---------------------------------------------------------------------------


def test_check_restart_token_returns_false_when_absent(tmp_path: Path) -> None:
    d = _make_daemon(tmp_path, "absent_daemon")
    token = tmp_path / "restart_tokens" / "absent_daemon.restart"
    with mock.patch.object(type(d), "_restart_token_path", new_callable=lambda: property(lambda self: token)):
        assert d._check_restart_token() is False


def test_check_restart_token_returns_true_when_present(tmp_path: Path) -> None:
    d = _make_daemon(tmp_path, "present_daemon")
    token = tmp_path / "restart_tokens" / "present_daemon.restart"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.touch()
    with mock.patch.object(type(d), "_restart_token_path", new_callable=lambda: property(lambda self: token)):
        assert d._check_restart_token() is True


# ---------------------------------------------------------------------------
# _consume_restart_token
# ---------------------------------------------------------------------------


def test_consume_restart_token_deletes_file(tmp_path: Path) -> None:
    d = _make_daemon(tmp_path, "consume_daemon")
    token = tmp_path / "restart_tokens" / "consume_daemon.restart"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.touch()
    with mock.patch.object(type(d), "_restart_token_path", new_callable=lambda: property(lambda self: token)):
        d._consume_restart_token()
    assert not token.exists()


def test_consume_restart_token_is_silent_when_already_gone(tmp_path: Path) -> None:
    d = _make_daemon(tmp_path, "gone_daemon")
    token = tmp_path / "restart_tokens" / "gone_daemon.restart"
    with mock.patch.object(type(d), "_restart_token_path", new_callable=lambda: property(lambda self: token)):
        d._consume_restart_token()  # must not raise


# ---------------------------------------------------------------------------
# run() restart-token integration
# ---------------------------------------------------------------------------


def test_run_exits_cleanly_when_token_present(tmp_path: Path) -> None:
    """run() must return (not loop) when a restart token exists at loop start."""
    d = _make_daemon(tmp_path, "exit_daemon")

    token_dir = _config.HUBRIS_HOME / "restart_tokens"
    token_path = token_dir / "exit_daemon.restart"
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path.touch()
    try:
        # run() should return without processing any actions.
        # Patch claim_next_memory_action to detect if it was ever called.
        with mock.patch("daemon_base._db.claim_next_memory_action") as mock_claim:
            d.run()
        mock_claim.assert_not_called()
    finally:
        token_path.unlink(missing_ok=True)


def test_run_consumes_token_on_exit(tmp_path: Path) -> None:
    """The token file must be deleted when run() exits due to it."""
    d = _make_daemon(tmp_path, "consume_on_exit")

    token_dir = _config.HUBRIS_HOME / "restart_tokens"
    token_path = token_dir / "consume_on_exit.restart"
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path.touch()
    try:
        d.run()
        assert not token_path.exists()
    finally:
        token_path.unlink(missing_ok=True)


def test_run_continues_when_no_token(tmp_path: Path) -> None:
    """run() must not exit early when no restart token is present."""
    d = _make_daemon(tmp_path, "continue_daemon")
    call_count = 0

    def _handler(root: Path, cfg: dict, subject_id: str | None, payload: dict) -> None:
        pass

    d.action_handlers["noop"] = _handler

    # Let the daemon process two iterations then set stop_requested.
    original_run_once = d.run_once

    def _counted_run_once(cfg: dict) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            d._stop_requested = True
        return original_run_once(cfg)

    with mock.patch.object(d, "run_once", side_effect=_counted_run_once):
        d.run()

    assert call_count >= 2, "run() exited before two iterations"
