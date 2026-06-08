"""supervisor.py - HuBrIS daemon supervisor.

Manages the lifecycle of background daemon subprocesses (daemon_watcher,
daemon_classify, etc.) as defined in manifest.json.  Provides:

  launch()   - start all daemons and kick off the health-check thread.
  shutdown() - terminate all running daemons.

Called once by server.py at startup after the user has saved config through
the startup UI.  All configuration is read from config.json.  Acquires
supervisor.lock to ensure only one supervisor manages the daemon cluster
at a time.
"""

import atexit
import json
import msvcrt
import os
import subprocess
import sys
import threading
from pathlib import Path

try:
    import psutil as _psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

import config as _config
from daemons import discover_daemon_specs
from log import get_logger

_log = get_logger("hubris.supervisor")

# Directory for per-daemon graceful restart tokens.
_RESTART_TOKENS_DIR = _config.HUBRIS_HOME / "restart_tokens"

# JSON file recording PIDs of daemons launched by the current supervisor,
# so a successor supervisor can kill orphans when stealing a dead lock.
_DAEMON_PIDS_PATH = _config.HUBRIS_HOME / "daemon_pids.json"

# name -> mtime (float) of the daemon's .py file at the time it was last started
_daemon_mtimes: dict[str, float] = {}

# name -> running Popen handle
_daemon_processes: dict[str, subprocess.Popen] = {}
_daemon_specs: list[dict] = []
_shutdown_event = threading.Event()


# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------

def _start_daemon(spec: dict) -> subprocess.Popen:
    """Launch a daemon subprocess for the given manifest spec."""
    module = spec["module"]
    name = spec["name"]
    cmd = [sys.executable, "-m", module]
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).parent,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _log.info("SUPERVISOR: started %s (module=%s, PID=%d)", name, module, proc.pid)
    # Record the module file's mtime so the health loop can detect code changes.
    daemon_file = Path(__file__).parent / f"{module}.py"
    if daemon_file.exists():
        _daemon_mtimes[name] = daemon_file.stat().st_mtime
    # Keep daemon_pids.json up to date so a successor supervisor can kill orphans.
    _write_daemon_pids()
    return proc


def _write_daemon_pids() -> None:
    """Persist the PIDs of all currently tracked daemon processes."""
    pids = {name: proc.pid for name, proc in _daemon_processes.items() if proc.poll() is None}
    try:
        _DAEMON_PIDS_PATH.write_text(json.dumps(pids))
    except OSError:
        pass


def _kill_orphan_daemons() -> None:
    """Kill any daemon PIDs recorded by a previous supervisor instance."""
    if not _DAEMON_PIDS_PATH.exists():
        return
    try:
        pids: dict = json.loads(_DAEMON_PIDS_PATH.read_text())
    except Exception:
        return
    for name, pid in pids.items():
        try:
            if _HAVE_PSUTIL:
                p = _psutil.Process(pid)
                p.terminate()
                try:
                    p.wait(timeout=3)
                except _psutil.TimeoutExpired:
                    p.kill()
            else:
                os.kill(pid, 9)
            _log.info("SUPERVISOR: killed orphan %s (PID=%d)", name, pid)
        except (ProcessLookupError, OSError):
            pass  # already gone
        except Exception as exc:
            _log.warning("SUPERVISOR: could not kill orphan %s (PID=%d): %s", name, pid, exc)
    try:
        _DAEMON_PIDS_PATH.unlink()
    except OSError:
        pass


def _health_check_loop() -> None:
    """Monitor daemon processes and restart any that have exited unexpectedly."""
    while not _shutdown_event.is_set():
        interval = min(
            (spec.get("health_check_interval_s", 5.0) for spec in _daemon_specs),
            default=5.0,
        )
        for spec in _daemon_specs:
            name = spec["name"]
            proc = _daemon_processes.get(name)
            if proc is not None and proc.poll() is not None:
                exit_code = proc.returncode
                _log.warning(
                    "SUPERVISOR: %s exited (code=%d) - restarting",
                    name, exit_code,
                )
                if spec.get("restart") == "always":
                    new_proc = _start_daemon(spec)  # _write_daemon_pids called inside
                    _daemon_processes[name] = new_proc
        # Write restart tokens for any running daemon whose module file has been
        # updated since it was last started.  The daemon's run() loop polls for
        # this token and exits cleanly; the crash-restart logic above then picks
        # it up on the next health-check cycle.
        for spec in _daemon_specs:
            name = spec["name"]
            proc = _daemon_processes.get(name)
            if proc is None or proc.poll() is not None:
                continue  # crashed or not running - crash-restart handles these
            daemon_file = Path(__file__).parent / f"{spec['module']}.py"
            if not daemon_file.exists():
                continue
            current_mtime = daemon_file.stat().st_mtime
            if current_mtime != _daemon_mtimes.get(name):
                token_path = _RESTART_TOKENS_DIR / f"{name}.restart"
                _RESTART_TOKENS_DIR.mkdir(parents=True, exist_ok=True)
                token_path.touch()
                _daemon_mtimes[name] = current_mtime
                _log.info(
                    "SUPERVISOR: %s code updated (mtime changed) - restart token written",
                    name,
                )
        _shutdown_event.wait(timeout=interval)


def shutdown() -> None:
    """Terminate all managed daemon processes."""
    _shutdown_event.set()
    for name, proc in list(_daemon_processes.items()):
        if proc.poll() is None:
            _log.info("SUPERVISOR: terminating %s (PID=%d)", name, proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    # Remove the PID registry so a successor does not try to kill already-dead processes.
    try:
        _DAEMON_PIDS_PATH.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Single-instance guard
# ---------------------------------------------------------------------------

_SUPERVISOR_LOCK_PATH = _config.HUBRIS_HOME / "supervisor.lock"
_supervisor_lock_fh = None


def _pid_is_alive(pid: int) -> bool:
    """Return True if the given PID is a running process."""
    if _HAVE_PSUTIL:
        return _psutil.pid_exists(pid)
    # Windows: os.kill(pid, 0) is unreliable (WinError 11 on Python 3.14 for
    # cross-architecture PIDs). Use OpenProcess with SYNCHRONIZE access instead.
    if sys.platform == "win32":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            # ERROR_ACCESS_DENIED (5) means the process exists but we can't open it.
            return ctypes.windll.kernel32.GetLastError() == 5
        except Exception:
            pass
    # POSIX fallback.
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _acquire_lock() -> bool:
    """
    Acquire supervisor.lock exclusively.  Returns True if this process is
    the primary supervisor, False if another live process already holds the lock.

    If the lock file exists but the recorded PID is no longer alive, the lock
    is considered stale.  We delete it, kill any orphan daemons recorded in
    daemon_pids.json, and then acquire the lock for ourselves.
    """
    global _supervisor_lock_fh
    _config.HUBRIS_HOME.mkdir(parents=True, exist_ok=True)

    # Check for a stale lock before trying to acquire.
    if _SUPERVISOR_LOCK_PATH.exists():
        try:
            raw = _SUPERVISOR_LOCK_PATH.read_bytes().strip()
            holder_pid = int(raw) if raw else 0
        except Exception:
            holder_pid = 0
        if holder_pid and not _pid_is_alive(holder_pid):
            _log.warning(
                "SUPERVISOR: lock held by dead PID %d - stealing lock and killing orphans",
                holder_pid,
            )
            _kill_orphan_daemons()
            try:
                _SUPERVISOR_LOCK_PATH.unlink()
            except OSError:
                pass

    try:
        _supervisor_lock_fh = open(_SUPERVISOR_LOCK_PATH, "a+b")
        msvcrt.locking(_supervisor_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        _supervisor_lock_fh.seek(0)
        _supervisor_lock_fh.truncate()
        _supervisor_lock_fh.write(str(os.getpid()).encode("ascii"))
        _supervisor_lock_fh.flush()

        def _release() -> None:
            try:
                msvcrt.locking(_supervisor_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                _supervisor_lock_fh.close()
            except OSError:
                pass

        atexit.register(_release)
        atexit.register(shutdown)
        return True
    except OSError:
        if _supervisor_lock_fh is not None:
            try:
                _supervisor_lock_fh.close()
            except OSError:
                pass
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def launch() -> bool:
    """
    Start all discovered daemons and kick off the health-check thread.

    All configuration (adapter, workspace_id, disabled daemons, etc.) is read
    from config.json, which the user saved through the startup UI before this
    is called.  No CLI arguments are forwarded to daemons.

    Returns True if this process acquired the supervisor lock and launched the
    daemons.  Returns False if another supervisor is already running; in that
    case server.py should serve MCP tools only.
    """
    global _daemon_specs
    if not _acquire_lock():
        _log.warning(
            "SUPERVISOR: another process holds %s - "
            "this instance will serve MCP tools but will not manage daemons",
            _SUPERVISOR_LOCK_PATH,
        )
        return False

    all_specs = discover_daemon_specs()
    disabled: list[str] = _config.load().get("disabled_daemons", [])
    for name in disabled:
        _log.warning("SUPERVISOR: daemon '%s' is disabled in config - will not launch", name)
    # Only manage daemons that are not in the disabled list. Filtering here
    # means the health-check loop is also unaware of disabled daemons.
    _daemon_specs = [s for s in all_specs if s["name"] not in disabled]
    # Clear restart tokens left by a previous server session before starting
    # any daemons.  A stale token would cause a daemon to exit immediately on
    # its first run-loop iteration.
    _RESTART_TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    for _stale_token in _RESTART_TOKENS_DIR.glob("*.restart"):
        try:
            _stale_token.unlink()
        except OSError:
            pass
    for spec in _daemon_specs:
        proc = _start_daemon(spec)
        _daemon_processes[spec["name"]] = proc

    health_thread = threading.Thread(
        target=_health_check_loop,
        name="supervisor-health",
        daemon=True,
    )
    health_thread.start()
    _log.info("SUPERVISOR: health-check thread started (PID %d)", os.getpid())
    return True
