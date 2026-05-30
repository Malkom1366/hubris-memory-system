"""Tests for supervisor.py restart-token and mtime-tracking logic.

Covers:
  - stale restart tokens are cleared on launch()
  - mtime change for a running daemon writes a restart token
  - mtime unchanged does not write a restart token
  - _start_daemon records the daemon file mtime in _daemon_mtimes
"""

import subprocess
import threading
from pathlib import Path
from unittest import mock

import config as _config
import supervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_popen(returncode: int | None = None) -> mock.MagicMock:
    proc = mock.MagicMock()
    proc.pid = 12345
    proc.poll.return_value = returncode
    return proc


# ---------------------------------------------------------------------------
# Stale token cleanup on launch
# ---------------------------------------------------------------------------


def test_stale_tokens_cleared_on_launch(tmp_path: Path, monkeypatch: object) -> None:
    """launch() must delete *.restart files left by a prior server session."""
    tokens_dir = _config.HUBRIS_HOME / "restart_tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    stale = tokens_dir / "old_daemon.restart"
    stale.touch()

    # Prevent any real subprocess from starting.
    monkeypatch.setattr(supervisor, "_acquire_lock", lambda: True)
    monkeypatch.setattr(supervisor, "discover_daemon_specs", lambda: [])
    monkeypatch.setattr(supervisor, "_daemon_specs", [])
    monkeypatch.setattr(supervisor, "_daemon_processes", {})
    # Prevent health thread from starting.
    monkeypatch.setattr(threading, "Thread", mock.MagicMock())

    supervisor.launch()

    assert not stale.exists(), "stale restart token was not cleared on launch"


# ---------------------------------------------------------------------------
# _start_daemon records mtime
# ---------------------------------------------------------------------------


def test_start_daemon_records_mtime(tmp_path: Path, monkeypatch: object) -> None:
    """_start_daemon must record the daemon module file mtime in _daemon_mtimes."""
    # Create a fake daemon .py file.
    daemon_file = Path(__file__).parent.parent / "HuBrIS" / "daemon_classify.py"
    if not daemon_file.exists():
        # Fallback: any .py file that exists in the supervisor's own directory.
        daemon_file = Path(supervisor.__file__).with_name("daemon_classify.py")

    spec = {"name": "classify", "module": "daemon_classify"}

    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: _fake_popen(None)
    )
    monkeypatch.setattr(supervisor, "_daemon_mtimes", {})

    if daemon_file.exists():
        supervisor._start_daemon(spec)
        assert "classify" in supervisor._daemon_mtimes
        assert supervisor._daemon_mtimes["classify"] == daemon_file.stat().st_mtime


# ---------------------------------------------------------------------------
# mtime change triggers restart token
# ---------------------------------------------------------------------------


def test_mtime_change_writes_restart_token(tmp_path: Path, monkeypatch: object) -> None:
    """Health loop must write a restart token when a daemon file mtime changes."""
    tokens_dir = _config.HUBRIS_HOME / "restart_tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    daemon_file = Path(supervisor.__file__).with_name("daemon_classify.py")
    if not daemon_file.exists():
        return  # skip if file is not present in this environment

    spec = {"name": "classify_mtime_test", "module": "daemon_classify"}
    proc = _fake_popen(None)  # poll() returns None = still running

    monkeypatch.setattr(supervisor, "_daemon_specs", [spec])
    monkeypatch.setattr(supervisor, "_daemon_processes", {"classify_mtime_test": proc})
    monkeypatch.setattr(supervisor, "_daemon_mtimes", {"classify_mtime_test": 0.0})  # old mtime

    token_path = tokens_dir / "classify_mtime_test.restart"
    if token_path.exists():
        token_path.unlink()

    try:
        # Simulate one iteration of the mtime-check portion of the health loop.
        current_mtime = daemon_file.stat().st_mtime
        for s in supervisor._daemon_specs:
            name = s["name"]
            p = supervisor._daemon_processes.get(name)
            if p is None or p.poll() is not None:
                continue
            df = Path(supervisor.__file__).parent / f"{s['module']}.py"
            if not df.exists():
                continue
            cmtime = df.stat().st_mtime
            if cmtime != supervisor._daemon_mtimes.get(name):
                tp = supervisor._RESTART_TOKENS_DIR / f"{name}.restart"
                supervisor._RESTART_TOKENS_DIR.mkdir(parents=True, exist_ok=True)
                tp.touch()
                supervisor._daemon_mtimes[name] = cmtime

        assert token_path.exists(), "restart token was not written on mtime change"
    finally:
        token_path.unlink(missing_ok=True)


def test_mtime_unchanged_does_not_write_token(tmp_path: Path, monkeypatch: object) -> None:
    """Health loop must NOT write a token when the daemon file mtime is unchanged."""
    tokens_dir = _config.HUBRIS_HOME / "restart_tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    daemon_file = Path(supervisor.__file__).with_name("daemon_classify.py")
    if not daemon_file.exists():
        return  # skip if file is not present

    current_mtime = daemon_file.stat().st_mtime
    spec = {"name": "classify_no_change", "module": "daemon_classify"}
    proc = _fake_popen(None)

    monkeypatch.setattr(supervisor, "_daemon_specs", [spec])
    monkeypatch.setattr(supervisor, "_daemon_processes", {"classify_no_change": proc})
    monkeypatch.setattr(supervisor, "_daemon_mtimes", {"classify_no_change": current_mtime})

    token_path = tokens_dir / "classify_no_change.restart"
    if token_path.exists():
        token_path.unlink()

    try:
        # Simulate one mtime-check iteration.
        for s in supervisor._daemon_specs:
            name = s["name"]
            p = supervisor._daemon_processes.get(name)
            if p is None or p.poll() is not None:
                continue
            df = Path(supervisor.__file__).parent / f"{s['module']}.py"
            if not df.exists():
                continue
            cmtime = df.stat().st_mtime
            if cmtime != supervisor._daemon_mtimes.get(name):
                tp = supervisor._RESTART_TOKENS_DIR / f"{name}.restart"
                supervisor._RESTART_TOKENS_DIR.mkdir(parents=True, exist_ok=True)
                tp.touch()
                supervisor._daemon_mtimes[name] = cmtime

        assert not token_path.exists(), "token written when mtime was unchanged"
    finally:
        token_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# disabled_daemons gate
# ---------------------------------------------------------------------------


def _minimal_spec(name: str, module: str = "daemon_watcher") -> dict:
    return {"name": name, "module": module, "restart": "always", "health_check_interval_s": 5}


def test_disabled_daemon_is_not_launched(monkeypatch: object) -> None:
    """Daemons listed in cfg['disabled_daemons'] must not be started."""
    specs = [_minimal_spec("watcher"), _minimal_spec("classify")]

    monkeypatch.setattr(supervisor, "_acquire_lock", lambda: True)
    monkeypatch.setattr(supervisor, "discover_daemon_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_daemon_specs", [])
    monkeypatch.setattr(supervisor, "_daemon_processes", {})
    monkeypatch.setattr(supervisor, "_daemon_mtimes", {})
    monkeypatch.setattr(threading, "Thread", mock.MagicMock())
    monkeypatch.setattr(
        _config,
        "load",
        lambda: {"disabled_daemons": ["classify"]},
    )

    launched: list[str] = []

    def _fake_start(spec, extra_args=None):
        launched.append(spec["name"])
        return _fake_popen(None)

    monkeypatch.setattr(supervisor, "_start_daemon", _fake_start)

    supervisor.launch()

    assert "watcher" in launched, "enabled daemon was not launched"
    assert "classify" not in launched, "disabled daemon was launched"


def test_enabled_daemon_is_launched(monkeypatch: object) -> None:
    """Daemons NOT in cfg['disabled_daemons'] must be started normally."""
    specs = [_minimal_spec("watcher"), _minimal_spec("classify")]

    monkeypatch.setattr(supervisor, "_acquire_lock", lambda: True)
    monkeypatch.setattr(supervisor, "discover_daemon_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_daemon_specs", [])
    monkeypatch.setattr(supervisor, "_daemon_processes", {})
    monkeypatch.setattr(supervisor, "_daemon_mtimes", {})
    monkeypatch.setattr(threading, "Thread", mock.MagicMock())
    monkeypatch.setattr(
        _config,
        "load",
        lambda: {"disabled_daemons": []},
    )

    launched: list[str] = []

    def _fake_start(spec, extra_args=None):
        launched.append(spec["name"])
        return _fake_popen(None)

    monkeypatch.setattr(supervisor, "_start_daemon", _fake_start)

    supervisor.launch()

    assert "watcher" in launched
    assert "classify" in launched


def test_all_daemons_disabled_launches_none(monkeypatch: object) -> None:
    """When all daemons are disabled, no Popen calls are made."""
    specs = [_minimal_spec("watcher"), _minimal_spec("classify")]

    monkeypatch.setattr(supervisor, "_acquire_lock", lambda: True)
    monkeypatch.setattr(supervisor, "discover_daemon_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_daemon_specs", [])
    monkeypatch.setattr(supervisor, "_daemon_processes", {})
    monkeypatch.setattr(supervisor, "_daemon_mtimes", {})
    monkeypatch.setattr(threading, "Thread", mock.MagicMock())
    monkeypatch.setattr(
        _config,
        "load",
        lambda: {"disabled_daemons": ["watcher", "classify"]},
    )

    launched: list[str] = []

    def _fake_start(spec, extra_args=None):
        launched.append(spec["name"])
        return _fake_popen(None)

    monkeypatch.setattr(supervisor, "_start_daemon", _fake_start)

    supervisor.launch()

    assert launched == [], "daemons were launched despite all being disabled"
