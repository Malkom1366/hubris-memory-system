"""
daemon_base.py - Base class for HuBrIS background action daemons.

Each daemon subclass (daemon_classify, daemon_watcher) inherits DaemonBase and
supplies a dict of action_type -> handler functions.  DaemonBase provides:

  run_once(cfg) -> bool
      Claim one pending action from the DB, dispatch to the handler, mark it
      complete or failed.  Returns True if an action was processed, False if the
      queue was empty.

  run()
      Poll loop: reload config each iteration (hot-reload), call run_once, sleep
      when idle, back off 5 s on errors, stop cleanly on SIGTERM / SIGBREAK.

Instantiate with:
    DaemonBase(
        name           - short label for log messages
        root           - Path to the HuBrIS memory root
        action_handlers - dict[str, Callable[[Path, dict, str|None, dict], None]]
        actor          - db actor constant (e.g. _db.ACTOR_MEMORY_ACTIONS)
        poll_interval  - seconds to sleep when the queue is empty (default 2.0)
    )
"""

import signal
import time
from pathlib import Path
from typing import Any

import config as _config
import db as _db
from log import get_logger

_log = get_logger("hubris.daemon_base")


class DaemonBase:
    """Minimal action-queue daemon: claim -> dispatch -> complete/fail loop."""

    def __init__(
        self,
        name: str,
        root: Path,
        action_handlers: dict[str, Any],
        actor: str,
        poll_interval: float = 2.0,
    ) -> None:
        self.name = name
        self.root = root
        self.action_handlers = action_handlers
        self.actor = actor
        self.poll_interval = poll_interval
        self._stop_requested = False
        self._install_signal_handlers()

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Register SIGTERM and (on Windows) SIGBREAK as stop signals."""
        def _handle_stop(signum: int, frame: Any) -> None:
            _log.info(
                "DAEMON %s: received signal %d - stopping after current action",
                self.name, signum,
            )
            self._stop_requested = True

        try:
            signal.signal(signal.SIGTERM, _handle_stop)
        except (OSError, ValueError):
            pass
        # SIGBREAK is Windows-only (Ctrl+Break).
        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _handle_stop)  # type: ignore[attr-defined]
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Restart token (graceful hot-restart coordination with supervisor)
    # ------------------------------------------------------------------

    @property
    def _restart_token_path(self) -> Path:
        return _config.HUBRIS_HOME / "restart_tokens" / f"{self.name}.restart"

    def _check_restart_token(self) -> bool:
        """Return True if a restart token exists for this daemon."""
        return self._restart_token_path.exists()

    def _consume_restart_token(self) -> None:
        """Delete the restart token so subsequent loop iterations do not exit again."""
        try:
            self._restart_token_path.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Single-action dispatch
    # ------------------------------------------------------------------

    def run_once(self, cfg: dict) -> bool:
        """
        Claim and process one action from the queue.

        Returns True if an action was processed (regardless of outcome),
        False if the queue is empty.
        """
        try:
            claimed = _db.claim_next_memory_action(
                self.actor, self.root,
                action_types=list(self.action_handlers.keys()),
            )
        except Exception as exc:
            _log.warning("DAEMON %s: claim failed: %s", self.name, exc)
            return False

        if claimed is None:
            return False

        action_id = int(claimed["id"])
        action_type = str(claimed["action_type"])
        subject_id = claimed.get("subject_id")
        payload = claimed.get("payload") or {}

        handler = self.action_handlers.get(action_type)
        if handler is None:
            msg = f"unknown action_type={action_type!r}"
            _log.warning("DAEMON %s action %s: %s", self.name, action_id, msg)
            _db.fail_memory_action(action_id, msg, self.actor, self.root)
            return True

        try:
            handler(self.root, cfg, subject_id, payload)
        except Exception as exc:
            new_status = _db.fail_memory_action(
                action_id, str(exc), self.actor, self.root
            )
            _log.warning(
                "DAEMON %s action %s (%s) failed (attempt %d, now %s): %s",
                self.name, action_id, action_type,
                claimed.get("attempts", 0), new_status, exc,
            )
            return True

        try:
            _db.complete_memory_action(action_id, self.actor, self.root)
            _log.info(
                "DAEMON %s action %s (%s) complete",
                self.name, action_id, action_type,
            )
        except Exception as exc:
            _log.warning(
                "DAEMON %s action %s: complete_memory_action failed: %s",
                self.name, action_id, exc,
            )
        return True

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main loop: reload config, process one action, sleep when idle.

        Stops when SIGTERM / SIGBREAK is received or _stop_requested is set.
        Backs off 5 s on unexpected errors to avoid tight crash loops.
        """
        _log.info("DAEMON %s: starting (root=%s)", self.name, self.root)
        while not self._stop_requested:
            if self._check_restart_token():
                self._consume_restart_token()
                _log.info(
                    "DAEMON %s: restart token detected - exiting for code update",
                    self.name,
                )
                return
            try:
                cfg = _config.load()
                processed = self.run_once(cfg)
                if not processed:
                    # Queue empty - wait before polling again.
                    time.sleep(self.poll_interval)
            except Exception as exc:
                _log.warning(
                    "DAEMON %s: unexpected error in run loop - backing off 5s: %s",
                    self.name, exc,
                )
                time.sleep(5.0)
        _log.info("DAEMON %s: stopped", self.name)
