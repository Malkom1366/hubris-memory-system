"""
daemon_synthesize.py - HuBrIS memory synthesis daemon.

Single job: drain finalize_subject actions from the memory_actions queue.

  finalize_subject - synthesise a definitive memory view for an archived subject

Run with: python -m daemon_synthesize
"""

import datetime
from pathlib import Path
from typing import Any

import config as _config
import db as _db
import meta_agent
import subjects as _subjects
from daemon_base import DaemonBase
from daemon_split import _maybe_enqueue_split
from log import get_logger

_log = get_logger("hubris.synthesize")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_finalize_subject(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Synthesise a definitive memory view for an archived subject.
    Reads the existing memory file for the current view and log, queries the DB
    for source messages ranked by relevance, and calls finalize_subject_memory()
    to produce a coherent final view from the bounded ranked messages.
    Post-synthesis: checks whether the newly written subject warrants a split.
    """
    if not subject_id:
        raise ValueError("finalize_subject action missing subject_id")

    all_subjects = _subjects.load_subjects(root)
    subject = next((s for s in all_subjects if s["id"] == subject_id), None)
    if subject is None:
        raise LookupError(f"subject id={subject_id[:8]} not found")

    subject_name = str(payload.get("subject_name") or subject.get("name") or "")
    _log.info("FINALIZE starting subject=%r id=%s", subject_name, subject_id[:8])

    existing_content = _subjects.read_subject_memory(subject, root)
    existing_view = _subjects.parse_memory_view(existing_content)
    event_log = _subjects.parse_memory_log(existing_content)

    ranked_messages = _db.get_subject_messages_for_finalize(subject_id, root)
    _log.info(
        "FINALIZE subject=%r id=%s -> %d ranked messages from DB",
        subject_name, subject_id[:8], len(ranked_messages),
    )

    view, log_entry = meta_agent.finalize_subject_memory(
        subject_name=subject_name,
        existing_view=existing_view,
        ranked_messages=ranked_messages,
        cfg=cfg,
    )

    if not view:
        raise RuntimeError(f"empty view returned for subject={subject_name!r}")

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    new_content = _subjects.compose_memory_file(
        subject_name=subject_name,
        view=view,
        new_log_entry=log_entry or "Final synthesis pass on archive.",
        status="revised",
        prior_log=event_log,
        timestamp=timestamp,
    )
    _subjects.write_subject_memory(subject, new_content, root)
    _log.info(
        "FINALIZE complete: subject=%r -> %d chars",
        subject_name, len(new_content),
    )
    # Record that synthesis has run so the scanner does not re-queue this subject.
    _db.mark_subject_synthesized(subject_id, _db.ACTOR_MEMORY_ACTIONS, root)
    # Post-finalize split check: the synthesised view may reveal that the
    # subject covers multiple distinct topics.
    _maybe_enqueue_split(subject, root, cfg)


def _handle_compact_memory(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Proactive context compaction.

    Archives subjects in LRU order (coldest first) and writes tombstones
    for messages that are now fully covered, stopping once the estimated
    freed-token count clears the configured threshold.
    """
    session_id = str(payload.get("session_id", ""))
    if not session_id:
        _log.warning("compact_memory: missing session_id in payload, skipping")
        return
    threshold = int(payload.get("threshold", _CONTEXT_COMPACT_THRESHOLD))

    lru_entries = _db.get_subjects_by_lru(session_id, root)
    if not lru_entries:
        _log.info(
            "compact_memory %s: no linked subjects found, nothing to compact",
            session_id[:8],
        )
        return

    closing_ids: set[str] = set()
    already_covered: set[int] = set()
    tombstone_map: dict[int, str] = {}
    freed_tokens = 0

    for entry in lru_entries:
        subj_id = str(entry["subject_id"])
        closing_ids.add(subj_id)

        try:
            subj = _subjects.get_subject(subj_id, root)
            subj_name = subj.get("name", subj_id) if subj else subj_id
            _subjects.set_subject_state(subj_id, "archived", root)
            _log.info(
                "compact_memory %s: archived subject %r (%s)",
                session_id[:8], subj_name, subj_id[:8],
            )
        except Exception as exc:
            _log.warning(
                "compact_memory: could not archive subject %s: %s",
                subj_id[:8] if len(subj_id) >= 8 else subj_id, exc,
            )
            subj_name = subj_id

        covered_now = (
            _db.get_messages_fully_covered(session_id, closing_ids, root)
            - already_covered
        )
        already_covered |= covered_now

        for msg_idx in sorted(covered_now):
            tombstone_map[msg_idx] = (
                f"[HUBRIS-COMPACT] Subject: {subj_name}. "
                f"Full content in long-term memory via recall_subject('{subj_id}')."
            )

        freed_tokens += len(covered_now) * 100

        if freed_tokens >= threshold and covered_now:
            break

    if not tombstone_map:
        _log.info("compact_memory %s: no messages became tombstonable", session_id[:8])
        return

    if _adapter is None:
        _log.warning(
            "compact_memory %s: no adapter available - tombstone write skipped",
            session_id[:8],
        )
        return

    _adapter.write_tombstones(session_id, tombstone_map)
    _log.info(
        "compact_memory %s: wrote %d tombstone(s), freed ~%d tokens",
        session_id[:8], len(tombstone_map), freed_tokens,
    )


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "finalize_subject": _handle_finalize_subject,
}

# ---------------------------------------------------------------------------
# Watch spec scanner (called by daemon_watcher on a schedule)
# ---------------------------------------------------------------------------


def _scan_archived_subjects_needing_synthesis(root: Path) -> list[dict]:
    """
    Return archived subjects that have not yet been synthesized
    (synthesized_at IS NULL).  Used as the recovery scanner so subjects
    archived during a server crash get a finalize_subject action on the
    next watcher cycle.

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    """
    try:
        subjects = _db.get_archived_subjects_needing_synthesis(root)
    except Exception as exc:
        _log.warning("SYNTHESIZE scan: DB query failed: %s", exc)
        return []
    return [
        {
            "subject_id": s["id"],
            "payload": {"subject_name": s["name"]},
        }
        for s in subjects
    ]


_WATCH_SPEC: list[dict] = [
    {
        "interval_s": 60,
        "action_type": "finalize_subject",
        "actor": _db.ACTOR_MEMORY_ACTIONS,
        "scanner": _scan_archived_subjects_needing_synthesis,
    }
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="synthesize",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
