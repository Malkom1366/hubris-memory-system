"""
daemon_watcher.py - HuBrIS session watcher daemon.

Runs as a subprocess managed by the server.py supervisor.  Owns the
SessionWatcher instance and all session-lifecycle logic:
  - Watches for session-file changes via SessionWatcher
  - On change: upserts messages to DB
  - Runs opportunistic vectorization passes
  - Manages subject dormancy and archiving lifecycle
  - Enforces session whitelist (opt-in) and workspace blacklist
(the supervisor health-check thread detects the exit and backs off).

Run with: python daemon_watcher.py [--adapter:<name>] [--session[:<id>]]
"""

import importlib
import json
import msvcrt
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import tiktoken

import config as _config
import db as _db
import subjects as _subjects
from daemon_split import _maybe_enqueue_split
from frontend_adapters import SessionAdapter
from log import get_logger
from watcher import SessionWatcher

_log = get_logger("hubris.watcher")

# ---------------------------------------------------------------------------
# Watch-spec discovery (Option B: module-attribute-based)
# ---------------------------------------------------------------------------


def _load_watch_specs(manifest_path: Path) -> list[dict]:
    """
    Read manifest.json and collect _WATCH_SPEC entries from every daemon module
    that declares one.  Each spec must have keys: interval_s, action_type,
    actor, scanner.  The watcher module itself is skipped to prevent a circular
    import.

    Returns a flat list of validated spec dicts ready for thread creation.
    """
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:
        _log.warning("WATCHER: could not read manifest for watch specs: %s", exc)
        return []

    specs: list[dict] = []
    for entry in manifest.get("daemons", []):
        module_name = entry.get("module", "")
        if not module_name or module_name == "daemon_watcher":
            continue
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            _log.debug("WATCHER: could not import %s for watch specs: %s", module_name, exc)
            continue
        watch_spec = getattr(mod, "_WATCH_SPEC", None)
        if watch_spec is None:
            continue
        for spec in watch_spec:
            required = ("interval_s", "action_type", "actor", "scanner")
            if not all(k in spec for k in required):
                _log.warning(
                    "WATCHER: malformed _WATCH_SPEC in %s (missing key): %s",
                    module_name, spec,
                )
                continue
            specs.append(spec)
            _log.info(
                "WATCHER: registered watch spec '%s' from %s (every %ds)",
                spec["action_type"], module_name, spec["interval_s"],
            )
    return specs


def _run_watch_spec_thread(
    spec: dict,
    root: Path,
    stop_event: threading.Event,
) -> None:
    """
    Periodic scan thread for a single _WATCH_SPEC entry.

    Waits `interval_s` before the first scan so restarts do not burst-enqueue
    immediately.  On each scan: calls spec["scanner"](root) to get a list of
    items, then enqueues any that do not already have a pending action.
    Exits cleanly when stop_event is set.
    """
    action_type: str = spec["action_type"]
    actor: str = spec["actor"]
    interval_s: float = float(spec["interval_s"])
    scanner = spec["scanner"]

    while not stop_event.is_set():
        # Wait first so we don't burst on every restart.
        stop_event.wait(timeout=interval_s)
        if stop_event.is_set():
            break
        try:
            items = scanner(root)
        except Exception as exc:
            _log.warning("WATCH SPEC %s: scanner error: %s", action_type, exc)
            continue
        enqueued = 0
        for item in items:
            subject_id: str | None = item.get("subject_id")
            payload: dict = item.get("payload") or {}
            try:
                if _db.has_pending_memory_action(action_type, subject_id, root):
                    continue
                _db.enqueue_memory_action(
                    action_type=action_type,
                    subject_id=subject_id,
                    payload=payload,
                    actor=actor,
                    root=root,
                )
                enqueued += 1
            except Exception as exc:
                _log.warning(
                    "WATCH SPEC %s: enqueue failed for subject_id=%s: %s",
                    action_type, subject_id, exc,
                )
        if enqueued:
            _log.info(
                "WATCH SPEC %s: enqueued %d action(s)",
                action_type, enqueued,
            )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TOKEN_ENCODING = tiktoken.get_encoding(_config.TOKEN_ENCODING_NAME)


def _count_session_tokens(messages: list[dict]) -> int:
    """Return the total cl100k_base token count for all message content strings."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if content:
            total += len(_TOKEN_ENCODING.encode(content))
    return total


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTEXT_COMPACT_THRESHOLD = 80_000

# ---------------------------------------------------------------------------
# Module-level state (all protected by _state_lock where mutated concurrently)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_session_message_counts: dict[str, int] = {}
_committed_counts: dict[str, int] = {}   # loaded at startup for restart recovery
_session_whitelist: set[str] = set()
_message_blacklist: dict[str, set[int]] = {}
_workspace_blacklist: set[str] = set()
_classify_failures: dict[str, int] = {}
_last_changed_session: str | None = None

# Session-binding state (set at startup from CLI args)
_bound_session: str | None = None

# The active SessionWatcher instance (set in main())
_watcher: SessionWatcher | None = None

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load_counts(root: Path) -> dict[str, int]:
    return _db.load_claimed_counts(root)


def _save_counts(root: Path, counts: dict[str, int]) -> None:
    try:
        _db.save_claimed_counts(counts, _db.ACTOR_SESSION_TRACKER, root)
    except Exception as exc:
        _log.warning("Could not persist session counts: %s", exc)


def _load_committed(root: Path) -> dict[str, int]:
    return _db.load_committed_counts(root)


def _save_committed(root: Path, counts: dict[str, int]) -> None:
    try:
        _db.save_committed_counts(counts, _db.ACTOR_SESSION_TRACKER, root)
    except Exception as exc:
        _log.warning("Could not persist committed counts: %s", exc)


def _load_session_whitelist(root: Path) -> set[str]:
    return _db.load_session_whitelist(root)


def _save_session_whitelist(root: Path, whitelist: set[str]) -> None:
    try:
        _db.save_session_whitelist(whitelist, _db.ACTOR_WHITELIST_MANAGER, root)
    except Exception as exc:
        _log.warning("Could not persist session whitelist: %s", exc)


def _load_message_blacklist(root: Path) -> dict[str, set[int]]:
    return _db.load_message_blacklist(root)


def _save_message_blacklist(root: Path, blacklist: dict[str, set[int]]) -> None:
    try:
        _db.save_message_blacklist(blacklist, _db.ACTOR_BLACKLIST_MANAGER, root)
    except Exception as exc:
        _log.warning("Could not persist message blacklist: %s", exc)


def _load_workspace_blacklist(root: Path) -> set[str]:
    return _db.load_workspace_blacklist(root)


def _save_workspace_blacklist(root: Path, blacklist: set[str]) -> None:
    try:
        _db.save_workspace_blacklist(blacklist, _db.ACTOR_BLACKLIST_MANAGER, root)
    except Exception as exc:
        _log.warning("Could not persist workspace blacklist: %s", exc)


def _load_classify_failures(root: Path) -> dict[str, int]:
    return _db.load_classify_failures(root)


def _save_classify_failures(root: Path, failures: dict[str, int]) -> None:
    try:
        _db.save_classify_failures(failures, _db.ACTOR_MESSAGE_CLASSIFIER, root)
    except Exception as exc:
        _log.warning("Could not persist classify failures: %s", exc)


# ---------------------------------------------------------------------------
# Dormant subject lifecycle
# ---------------------------------------------------------------------------


def _check_dormant_subjects(root: Path, cfg: dict) -> None:
    """
    Promote open subjects to dormant and dormant subjects to archived based
    on elapsed time since last_activity.  Rebuilds and broadcasts the catalog
    anchor if any transitions occurred.  Also runs the watchdog split detector
    on all non-archived subjects.

    Delegates the lifecycle pass to subjects.run_lifecycle_pass so the
    threshold logic lives in exactly one place.

    Called at the end of every _on_session_changed cycle.
    All errors are logged; none propagate.
    """
    _subjects.run_lifecycle_pass(root, cfg, _watcher.adapter)

    # Watchdog split detector: scan all open/dormant subjects for threshold breaches.
    all_subjects = _subjects.load_subjects(root)
    for s in all_subjects:
        if s["state"] == "archived":
            continue
        try:
            _maybe_enqueue_split(s, root, cfg)
        except Exception as exc:
            _log.warning("AUTO-SPLIT check failed for subject=%r: %s", s.get("name"), exc)


# ---------------------------------------------------------------------------
# Vectorization pass
# ---------------------------------------------------------------------------
# Session change handler
# ---------------------------------------------------------------------------


def _on_session_changed(session_id: str, adapter: SessionAdapter) -> None:
    """
    Invoked by the watcher when a session file changes.
    Processes the delta: classify, update memory, check/restore anchor.
    """
    # Whitelist check: only process sessions the user has explicitly opted in
    # via the config Sessions tab.
    if session_id not in _session_whitelist:
        return

    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    workspace_id = adapter.get_session_workspace(session_id)

    # Skip sessions belonging to blacklisted workspaces.
    if workspace_id is not None and workspace_id in _workspace_blacklist:
        return

    # If the adapter's workspace filter rejects this workspace, skip silently.
    if not adapter.workspace_id_allowed(workspace_id, cfg):
        return

    # Explicit session binding (--session:<id> CLI flag): only process the bound session.
    if _bound_session is not None:
        if session_id != _bound_session:
            return

    messages = adapter.read_messages(session_id)
    if not messages:
        return

    _db.upsert_session_messages(
        session_id,
        messages,
        cfg.get("workspace_id", "global"),
        _db.ACTOR_MEMORY_WRITER,
        root,
    )

    # Atomically claim the new message range - read and advance the count in
    # one lock acquisition so concurrent watcher events for the same session
    # (e.g. anchor refresh fires immediately after a Continue write) cannot
    # both see prev=0 and double-launch CLASSIFY.
    global _last_changed_session
    with _state_lock:
        prev_count = _session_message_counts.get(session_id, 0)
        new_count = len(messages)
        if new_count <= prev_count:
            return
        _session_message_counts[session_id] = new_count
        _last_changed_session = session_id
        counts_snapshot = dict(_session_message_counts)

    _save_counts(root, counts_snapshot)
    new_messages = messages[prev_count:]

    _log.info(
        "SESSION CHANGED %s: delta=%d new msg(s) (total=%d, prev=%d)",
        session_id[:8], len(new_messages), new_count, prev_count,
    )

    # Proactive context compaction: enqueue compact_memory if the session is
    # growing large enough to risk GHCP auto-compaction at the model limit.
    _compact_threshold = int(cfg.get("context_compact_threshold", _CONTEXT_COMPACT_THRESHOLD))
    if (
        _count_session_tokens(messages) >= _compact_threshold
        and not _db.has_pending_memory_action("compact_memory", None, root)
    ):
        _db.enqueue_memory_action(
            action_type="compact_memory",
            subject_id=None,
            payload={"session_id": session_id, "threshold": _compact_threshold},
            actor=_db.ACTOR_MEMORY_WRITER,
            root=root,
        )
        _log.info(
            "ENQUEUE COMPACT %s: session token count >= %d",
            session_id[:8], _compact_threshold,
        )

    # Per-adapter context push: anchor refresh (Continue) or fence injection (GHCP).
    adapter.push_context(session_id, messages, prev_count, cfg, root)

    # Autonomous lifecycle: promote stale open subjects to dormant and archive
    # subjects that have been dormant long enough.
    try:
        _check_dormant_subjects(root, cfg)
    except Exception as exc:
        _log.warning("DORMANT CHECK unexpected error: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global _session_message_counts, _committed_counts
    global _session_whitelist, _message_blacklist, _workspace_blacklist, _classify_failures
    global _bound_session, _watcher

    cfg = _config.load()

    # Parse CLI args via the shared config.parse_startup_args helper.
    argv_adapter, argv_session_id, _configure = _config.parse_startup_args(sys.argv[1:])

    if argv_adapter:
        cfg["adapter"] = argv_adapter
        _log.info("WATCHER DAEMON: adapter overridden via CLI -> '%s'", argv_adapter)
    if argv_session_id:
        _bound_session = argv_session_id
        _log.info("WATCHER DAEMON: session bound via CLI -> %s", argv_session_id[:8])

    root = _config.memory_root(cfg.get("workspace_id", "global"))

    # Load all persisted state from DB.
    _committed_counts = _load_committed(root)
    _session_message_counts = _load_counts(root)
    _session_whitelist = _load_session_whitelist(root)
    _message_blacklist = _load_message_blacklist(root)
    _workspace_blacklist = _load_workspace_blacklist(root)
    _classify_failures = _load_classify_failures(root)

    _watcher = SessionWatcher(
        on_session_changed=_on_session_changed,
        cfg=cfg,
        status_file=_config.HUBRIS_HOME / "watcher_status.json",
    )

    # Restart recovery: if claimed > committed for any session, classify was
    # interrupted. Reset claimed to committed so the next watcher event retries.
    recovered = 0
    for sid in list(_session_message_counts.keys()):
        committed = _committed_counts.get(sid, 0)
        claimed = _session_message_counts[sid]
        if claimed > committed:
            _log.info(
                "RESTART RECOVERY %s: claimed=%d committed=%d -> will retry %d msg(s)",
                sid[:8], claimed, committed, claimed - committed,
            )
            _session_message_counts[sid] = committed
            recovered += 1
    if recovered:
        _save_counts(root, _session_message_counts)
        _log.info("RESTART RECOVERY complete: %d session(s) rolled back for retry", recovered)
    elif _session_message_counts:
        _log.info(
            "Loaded %d persisted session count(s) (all committed - no recovery needed)",
            len(_session_message_counts),
        )

    # Single-instance guard: only one watcher daemon may run at a time.
    # If another process holds the lock we exit; the supervisor health-check
    # detects the exit code and backs off before trying to restart.
    watcher_lock_path = _config.HUBRIS_HOME / "watcher.lock"
    _config.HUBRIS_HOME.mkdir(parents=True, exist_ok=True)
    watcher_lock_fh = None
    try:
        watcher_lock_fh = open(watcher_lock_path, "a+b")
        msvcrt.locking(watcher_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        watcher_lock_fh.seek(0)
        watcher_lock_fh.truncate()
        watcher_lock_fh.write(str(os.getpid()).encode("ascii"))
        watcher_lock_fh.flush()
    except OSError:
        _log.warning(
            "WATCHER DAEMON: watcher.lock held by another process - exiting (exit code 1)"
        )
        if watcher_lock_fh is not None:
            try:
                watcher_lock_fh.close()
            except OSError:
                pass
        sys.exit(1)

    _watcher.start()
    _log.info("WATCHER DAEMON: started (PID %d)", os.getpid())

    # Stop event and signal handlers must be set up BEFORE watch-spec threads are
    # started, because those threads receive stop_event as an argument.
    stop_event = threading.Event()

    def _handle_stop(signum: int, frame: Any) -> None:
        _log.info("WATCHER DAEMON: received signal %d - stopping", signum)
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, _handle_stop)
    except (OSError, ValueError):
        pass
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _handle_stop)  # type: ignore[attr-defined]
        except (OSError, ValueError):
            pass

    # Discover and start periodic watch-spec threads from daemon _WATCH_SPEC attributes.
    _manifest_path = Path(__file__).parent / "manifest.json"
    _watch_specs = _load_watch_specs(_manifest_path)
    for _spec in _watch_specs:
        _t = threading.Thread(
            target=_run_watch_spec_thread,
            args=(_spec, root, stop_event),
            daemon=True,
            name=f"watch-spec-{_spec['action_type']}",
        )
        _t.start()
        _log.info(
            "WATCHER DAEMON: started watch-spec thread '%s' (every %ds)",
            _spec["action_type"], _spec["interval_s"],
        )

    stop_event.wait()

    _log.info("WATCHER DAEMON: stopping")
    try:
        msvcrt.locking(watcher_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
        watcher_lock_fh.close()
    except OSError:
        pass


if __name__ == "__main__":
    main()
