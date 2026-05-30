"""
daemon_classify.py - HuBrIS message classification daemon.

Single job: drain classify_message and classify_memory actions from the
memory_actions queue.

  classify_message  - classify a session message delta into memory subjects
  classify_memory   - re-attribute messages from a split parent to children

Also exports _WATCH_SPEC so daemon_watcher discovers this daemon's scanner at
startup and periodically enqueues classify_message actions for sessions whose
AB record count exceeds the committed classify HWM.

Module-level state (loaded from DB at startup, mutated by handlers):
  _state_lock         - protects _committed_counts, _classify_failures, _message_blacklist
  _committed_counts   - {session_id: last committed classify HWM}
  _classify_failures  - {"session_id:offset": failure_count}
  _message_blacklist  - {session_id: {message_index: reason}} for blacklisted messages

Run with: python -m daemon_classify [--adapter:<name>]
"""

import datetime
import sys
import threading
from pathlib import Path
from typing import Any

import config as _config
import db as _db
import meta_agent
import subjects as _subjects
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.classify")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLASSIFY_FAILURE_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Module-level state (loaded at startup, mutated by handlers)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_committed_counts: dict[str, int] = {}
_classify_failures: dict[str, int] = {}
_message_blacklist: dict[str, dict[int, str]] = {}

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _assignments_dir(root: Path) -> Path:
    d = Path(root) / "assignments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_assignments(root: Path, session_id: str) -> dict[int, str | None]:
    return _db.load_assignments(session_id, root)


def _save_assignments(
    root: Path,
    session_id: str,
    assignments: dict[int, str | None],
) -> None:
    try:
        _db.save_assignments(session_id, assignments, _db.ACTOR_MESSAGE_CLASSIFIER, root)
    except Exception as exc:
        _log.warning("Could not persist assignments for %s: %s", session_id[:8], exc)


def _save_committed(root: Path, counts: dict[str, int]) -> None:
    try:
        _db.save_committed_counts(counts, _db.ACTOR_SESSION_TRACKER, root)
    except Exception as exc:
        _log.warning("Could not persist committed counts: %s", exc)


def _save_classify_failures(root: Path, failures: dict[str, int]) -> None:
    try:
        _db.save_classify_failures(failures, _db.ACTOR_MESSAGE_CLASSIFIER, root)
    except Exception as exc:
        _log.warning("Could not persist classify failures: %s", exc)


def _save_message_blacklist(root: Path, blacklist: dict[str, dict[int, str]]) -> None:
    try:
        _db.save_message_blacklist(blacklist, _db.ACTOR_BLACKLIST_MANAGER, root)
    except Exception as exc:
        _log.warning("Could not persist message blacklist: %s", exc)


# ---------------------------------------------------------------------------
# Memory action handlers
# ---------------------------------------------------------------------------


def _handle_classify_message(
    root: Path,
    cfg: dict,
    subject_id: str | None,  # always None for classify_message actions
    payload: dict,
) -> None:
    """
    Classify a delta of new messages from a single session into memory subjects
    (Stage 1 fast classification). Null-assigned messages are collected and
    handed off to daemon_escalate via an escalate_message queue action.

    On ClassifyFailed below the auto-blacklist threshold the handler re-raises so
    the queue marks the action failed and retries it up to max_memory_action_attempts.
    """
    session_id: str = payload["session_id"]
    from_index: int = int(payload["from_index"])
    to_index: int = int(payload["to_index"])

    # Load all messages for the session.
    raw_rows = _db.get_all_messages_for_session(session_id, root)
    messages: list[dict] = raw_rows  # already has 'content' and 'role' keys

    if not messages:
        _log.warning(
            "CLASSIFY_MESSAGE %s: no messages found in DB - action abandoned",
            session_id[:8],
        )
        return

    new_messages = messages[from_index:to_index]
    if not new_messages:
        _log.info(
            "CLASSIFY_MESSAGE %s: delta %d..%d is empty - skipping",
            session_id[:8], from_index, to_index,
        )
        return

    known = _subjects.load_subjects(root)
    _classify_assigned_total = 0
    _subject_message_buffer: dict[str, list[dict]] = {}
    _session_assignments: dict[int, str | None] = _load_assignments(root, session_id)
    _session_relevance: dict[int, dict[str, float]] = {}

    def _on_batch_done(batch_relevance: dict[int, dict[str, float]], messages_cumulative: int) -> None:
        nonlocal _classify_assigned_total
        for global_idx, pair_map in batch_relevance.items():
            if not pair_map:
                continue
            if not (0 <= global_idx < len(new_messages)):
                continue
            for subj_id, score in pair_map.items():
                try:
                    _subjects.increment_message_count(subj_id, delta=1, root=root)
                    _classify_assigned_total += 1
                except Exception:
                    pass
                _subject_message_buffer.setdefault(subj_id, []).append(
                    {**new_messages[global_idx], "relevance": float(score)}
                )
        batch_relevance_abs: dict[int, dict[str, float]] = {}
        for global_idx, pair_map in batch_relevance.items():
            abs_idx = from_index + global_idx
            if pair_map:
                primary = max(pair_map.items(), key=lambda kv: kv[1])[0]
                _session_assignments[abs_idx] = primary
            else:
                _session_assignments[abs_idx] = None
            _session_relevance[abs_idx] = dict(pair_map)
            batch_relevance_abs[abs_idx] = dict(pair_map)
        _save_assignments(root, session_id, batch_relevance_abs)
        with _state_lock:
            _committed_counts[session_id] = from_index + messages_cumulative
            snapshot = dict(_committed_counts)
        _save_committed(root, snapshot)

    if known:
        try:
            meta_agent.classify_messages(new_messages, known, cfg, on_batch_done=_on_batch_done)
        except meta_agent.ClassifyFailed as exc:
            with _state_lock:
                committed_now = _committed_counts.get(session_id, 0)
                failure_key = f"{session_id}:{committed_now}"
                _classify_failures[failure_key] = _classify_failures.get(failure_key, 0) + 1
                fail_count = _classify_failures[failure_key]
                failures_snapshot = dict(_classify_failures)
            _save_classify_failures(root, failures_snapshot)
            threshold = int(cfg.get("classify_failure_threshold", _CLASSIFY_FAILURE_THRESHOLD))

            if fail_count >= threshold:
                # Auto-blacklist path: advance past the bad batch, complete the
                # action (no retry needed - further messages will use a fresh action).
                # reason reflects the nature of the final failure so the entry
                # can be evaluated for retry (timeout = infrastructure, recoverable;
                # content_failure = LLM could not parse, less likely to recover).
                reason = "timeout" if exc.is_transient else "content_failure"
                batch_size = max(1, int(cfg.get("classify_batch_size", meta_agent._CLASSIFY_BATCH_SIZE)))
                bl_end = committed_now + batch_size
                with _state_lock:
                    if session_id not in _message_blacklist:
                        _message_blacklist[session_id] = {}
                    _message_blacklist[session_id].update(
                        {idx: reason for idx in range(committed_now, bl_end)}
                    )
                    _committed_counts[session_id] = bl_end
                    committed_snapshot = dict(_committed_counts)
                    del _classify_failures[failure_key]
                    failures_snapshot = dict(_classify_failures)
                    msg_bl_snapshot = {k: dict(v) for k, v in _message_blacklist.items()}
                _save_committed(root, committed_snapshot)
                _save_classify_failures(root, failures_snapshot)
                _save_message_blacklist(root, msg_bl_snapshot)
                _log.warning(
                    "AUTO_BLACKLIST %s: messages %d-%d blacklisted (reason=%s) after %d consecutive failures",
                    session_id[:8], committed_now, bl_end - 1, reason, fail_count,
                )
                return  # action completes normally; no retry
            else:
                remaining = to_index - committed_now
                _log.warning(
                    "CLASSIFY_FAILED %s: batch failed (attempt %d/%d) - committed=%d, "
                    "%d msg(s) will retry via queue",
                    session_id[:8], fail_count, threshold, committed_now, remaining,
                )
                raise  # queue retries this action
    else:
        # No subjects exist yet. Record all new messages as null-assigned and
        # advance the HWM so the null pool grows across turns for later escalation.
        no_links: dict[int, dict[str, float]] = {}
        for i in range(len(new_messages)):
            abs_idx = from_index + i
            _session_assignments[abs_idx] = None
            _session_relevance[abs_idx] = {}
            no_links[abs_idx] = {}
        _save_assignments(root, session_id, no_links)
        with _state_lock:
            _committed_counts[session_id] = from_index + len(new_messages)
            snapshot = dict(_committed_counts)
        _save_committed(root, snapshot)
        _log.info(
            "CLASSIFY SKIP %s: no subjects yet - %d msg(s) added to null pool",
            session_id[:8], len(new_messages),
        )

    _log.info(
        "ZERO-INFERENCE UPDATE %s: %d subject(s) incremented",
        session_id[:8], _classify_assigned_total,
    )

    # Enqueue any null-assigned messages from this delta for escalation.
    null_ids = [
        new_messages[i]["id"]
        for i in range(len(new_messages))
        if _session_assignments.get(from_index + i) is None
    ]
    if null_ids:
        _db.enqueue_memory_action(
            action_type="escalate_message",
            subject_id=None,
            payload={"session_id": session_id, "message_ids": null_ids},
            actor=_db.ACTOR_MESSAGE_CLASSIFIER,
            root=root,
        )
        _log.info(
            "ESCALATE ENQUEUE %s: %d null msg(s) handed off to daemon_escalate",
            session_id[:8], len(null_ids),
        )

    # Write updated memory files for every subject that received new messages.
    if _subject_message_buffer:
        relevance_floor = float(cfg.get("relevance_floor", 0.2))
        _log.info(
            "MEMORY WRITE %s: updating %d subject(s) (relevance_floor=%.2f)",
            session_id[:8], len(_subject_message_buffer), relevance_floor,
        )
        all_subjects = _subjects.load_subjects(root)
        subject_lookup = {s["id"]: s for s in all_subjects}
        timestamp = datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S")
        for subj_id, msgs in _subject_message_buffer.items():
            subject = subject_lookup.get(subj_id)
            if subject is None:
                continue
            filtered_msgs = [
                m for m in msgs
                if float(m.get("relevance", 1.0)) >= relevance_floor
            ]
            if not filtered_msgs:
                _log.info(
                    "MEMORY WRITE %s: subject=%r all %d msg(s) below floor - skipped",
                    session_id[:8], subject["name"], len(msgs),
                )
                continue
            try:
                existing_content = _subjects.read_subject_memory(subject, root)
                existing_view = _subjects.parse_memory_view(existing_content)
                prior_log = _subjects.parse_memory_log(existing_content)
                view, log_entry, status = meta_agent.update_subject_memory(
                    subject_name=subject["name"],
                    existing_view=existing_view,
                    new_messages=filtered_msgs,
                    cfg=cfg,
                )
                if view:
                    new_content = _subjects.compose_memory_file(
                        subject_name=subject["name"],
                        view=view,
                        new_log_entry=log_entry,
                        status=status,
                        prior_log=prior_log,
                        timestamp=timestamp,
                    )
                    _subjects.write_subject_memory(subject, new_content, root)
                    _log.info(
                        "MEMORY WRITE %s: subject=%r %d msg(s) (of %d) status=%s -> %d chars",
                        session_id[:8], subject["name"], len(filtered_msgs),
                        len(msgs), status, len(new_content),
                    )
            except Exception as exc:
                _log.warning(
                    "MEMORY WRITE %s: subject=%r failed (%s)",
                    session_id[:8], subject.get("name", subj_id),
                    type(exc).__name__,
                )


# ---------------------------------------------------------------------------
# Watch spec scanner
# ---------------------------------------------------------------------------


def _scan_unclassified_sessions(root: Path) -> list[dict]:
    """
    Return classify_message items for sessions whose AB record count exceeds
    the committed classify HWM.

    Each item shape:
      subject_id - session_id, used as the dedup key in memory_actions
      payload    - all data the handler needs: session_id, from_index, to_index

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    """
    try:
        committed = _db.load_committed_counts(root)
        totals = _db.get_session_message_counts(root)
    except Exception as exc:
        _log.warning("CLASSIFY scan: DB query failed: %s", exc)
        return []

    items = []
    for session_id, total in totals.items():
        hwm = committed.get(session_id, 0)
        if total > hwm:
            items.append({
                "subject_id": session_id,
                "payload": {
                    "session_id": session_id,
                    "from_index": hwm,
                    "to_index": total,
                },
            })
    return items


# ---------------------------------------------------------------------------
# Watch spec (read by daemon_watcher at startup via _WATCH_SPEC discovery)
# ---------------------------------------------------------------------------

_WATCH_SPEC: list[dict[str, Any]] = [
    {
        "interval_s": 30,
        "action_type": "classify_message",
        "actor": _db.ACTOR_MESSAGE_CLASSIFIER,
        "scanner": _scan_unclassified_sessions,
    }
]


# ---------------------------------------------------------------------------
# Action handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "classify_message": _handle_classify_message,
}


# ---------------------------------------------------------------------------
# Startup state loader
# ---------------------------------------------------------------------------


def _setup_classify_state(root: Path, cfg: dict) -> None:
    """
    Load persisted state from DB.
    Called once at daemon startup before the run loop begins.
    """
    global _committed_counts, _classify_failures, _message_blacklist

    _committed_counts = _db.load_committed_counts(root)
    _classify_failures = _db.load_classify_failures(root)
    _message_blacklist = _db.load_message_blacklist(root)

    _log.info(
        "CLASSIFY DAEMON state loaded: committed=%d sessions, failures=%d, bl_ranges=%d",
        len(_committed_counts), len(_classify_failures), len(_message_blacklist),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    _setup_classify_state(root, cfg)

    daemon = DaemonBase(
        name="classify",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
