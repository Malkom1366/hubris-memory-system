"""
Session watcher for HuBrIS.

Watches ALL Continue session files simultaneously using a background thread
that stats each file every poll_interval seconds.

Design rules:
- Uses os.path.getmtime (stat-based), not filesystem events. More predictable
  on Windows NTFS and works with network drives.
- Maintains a dict of {session_id: last_mtime}.
- New session files are discovered automatically on each poll cycle.
- sessions.json (the Continue index file) is always excluded.
- If a meta-agent cycle is already running for a session, the next change is
  queued once; subsequent changes before the cycle finishes are deduped.
- Only one executor thread slot is used (max_workers=1) to prevent runaway
  LLM calls when multiple sessions change simultaneously. This means changes
  are processed one at a time in arrival order.
"""

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import config as _config
from frontend_adapters import SessionAdapter, build_active_frontend_adapters


class SessionWatcher:
    """
    Background watcher that detects changes to Continue session files and
    calls a callback for each changed session.

    The callback receives (session_id: str, adapter: SessionAdapter).
    It is invoked on the executor thread (not the poll thread).
    """

    def __init__(
        self,
        adapter: SessionAdapter | list[SessionAdapter] | None = None,
        on_session_changed: Callable[[str, SessionAdapter], None] | None = None,
        cfg: dict[str, Any] | None = None,
        status_file: Path | None = None,
    ) -> None:
        if cfg is None:
            cfg = _config.load()
        self._cfg = cfg
        if adapter is not None:
            if isinstance(adapter, list):
                self._adapters: list[SessionAdapter] = adapter
            else:
                self._adapters = [adapter]
        else:
            self._adapters = build_active_frontend_adapters(cfg)
        self._callback = on_session_changed
        self._poll_interval: float = float(cfg.get("poll_interval", 2))
        self._status_file: Path | None = status_file

        # session_id -> last known mtime
        self._watched: dict[str, float] = {}
        # session_id -> which adapter owns this session
        self._session_adapter_map: dict[str, SessionAdapter] = {}
        # session_id -> pending Future (if a cycle is in flight)
        self._in_flight: dict[str, Future] = {}
        # session_id -> queued=True (next change to process after in-flight finishes)
        self._queued: dict[str, bool] = {}

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hubris-bg")

    # ------------------------------------------------------------------
    # Adapter access
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> SessionAdapter:
        """Return the primary frontend adapter (first in the active adapters list).

        Single-adapter convenience accessor. Code that operates on a specific
        session should use _session_adapter_map[session_id] once multi-adapter
        support is fully active.
        """
        return self._adapters[0]

    def get_adapters(self) -> list[SessionAdapter]:
        """Return all active adapters managed by this watcher."""
        return list(self._adapters)

    @property
    def _adapter(self) -> SessionAdapter:
        """Deprecated - use .adapter instead."""
        return self.adapter

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background poll thread. Idempotent."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="hubris-watcher",
            daemon=True,
        )
        self._poll_thread.start()

    def stop(self) -> None:
        """Signal the poll thread to exit and wait for it."""
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=10)
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:
                # Never let a transient error kill the watcher thread.
                pass
            self._write_status_file()
            self._stop_event.wait(timeout=self._poll_interval)

    def _write_status_file(self) -> None:
        """Write the current watcher status to self._status_file as JSON (no-op if unset)."""
        if self._status_file is None:
            return
        data = self.status()
        data["timestamp"] = time.time()
        try:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            self._status_file.write_text(json.dumps(data))
        except OSError:
            pass

    def _poll_once(self) -> None:
        """Stat all known and newly discovered session files."""
        # Collect current sessions from all adapters; track which adapter owns each.
        new_adapter_map: dict[str, SessionAdapter] = {}
        for adapter in self._adapters:
            for sid in adapter.list_sessions():
                new_adapter_map[sid] = adapter
        current_sessions = set(new_adapter_map)

        with self._lock:
            # Refresh the adapter ownership map.
            self._session_adapter_map = new_adapter_map

            # Discover new sessions.
            for sid in current_sessions:
                if sid not in self._watched:
                    self._watched[sid] = 0.0  # force process on first sight

            # Check for changes.
            changed: list[str] = []
            for sid in list(self._watched):
                if sid not in current_sessions:
                    # Session file removed; stop watching.
                    del self._watched[sid]
                    continue
                adapter = self._session_adapter_map.get(sid)
                if adapter is None:
                    continue
                try:
                    mtime = adapter.get_mtime(sid)
                except OSError:
                    continue
                if mtime != self._watched[sid]:
                    self._watched[sid] = mtime
                    changed.append(sid)

        for sid in changed:
            self._enqueue(sid)

    # ------------------------------------------------------------------
    # Work queueing
    # ------------------------------------------------------------------

    def _enqueue(self, session_id: str) -> None:
        """
        Submit a meta-agent cycle for session_id, or mark it queued if one
        is already in flight.
        """
        if self._callback is None:
            return
        with self._lock:
            existing = self._in_flight.get(session_id)
            if existing is not None and not existing.done():
                # Already running - queue the next run.
                self._queued[session_id] = True
                return
            # Submit immediately.
            future = self._executor.submit(self._run_cycle, session_id)
            self._in_flight[session_id] = future

    def _run_cycle(self, session_id: str) -> None:
        """Execute the meta-agent callback, then flush the queue if needed."""
        # Capture the adapter reference under the lock before doing any I/O.
        with self._lock:
            adapter = self._session_adapter_map.get(session_id)
        try:
            if self._callback and adapter is not None:
                self._callback(session_id, adapter)
        finally:
            # After the callback returns, absorb any mtime changes that HuBrIS
            # itself made (anchor refresh, catalog broadcast, synthesis writes).
            # Reading mtime outside the lock avoids blocking I/O while locked.
            new_mtime: float | None = None
            try:
                if adapter is not None:
                    new_mtime = adapter.get_mtime(session_id)
            except OSError:
                pass

            with self._lock:
                if new_mtime is not None and session_id in self._watched:
                    self._watched[session_id] = new_mtime
                self._in_flight.pop(session_id, None)
                if self._queued.pop(session_id, False):
                    # There was a pending change; fire again.
                    future = self._executor.submit(self._run_cycle, session_id)
                    self._in_flight[session_id] = future

    # ------------------------------------------------------------------
    # Config reload
    # ------------------------------------------------------------------

    def update_poll_interval(self, seconds: float) -> None:
        """Hot-reload the poll interval. Takes effect on the next sleep cycle."""
        self._poll_interval = seconds

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "watching": list(self._watched.keys()),
                "session_count": len(self._watched),
                "in_flight": list(self._in_flight.keys()),
                "queued": [k for k, v in self._queued.items() if v],
                "poll_interval": self._poll_interval,
                "running": self._poll_thread is not None and self._poll_thread.is_alive(),
            }
