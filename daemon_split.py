"""
daemon_split.py - HuBrIS subject split daemon.

Single job: drain split_subject actions from the memory_actions queue.

  split_subject  - partition an oversized subject into 2-5 child subjects,
                   create each child's memory view, archive the parent, and
                   enqueue a classify_memory action to re-attribute messages.

Run with: python -m daemon_split
"""

import datetime
import sys
from pathlib import Path
from typing import Any

import config as _config
import db as _db
import meta_agent
import subjects as _subjects
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.split")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Children of a split subject may themselves be split; grandchildren may not.
_SPLIT_MAX_DEPTH = 1

# ---------------------------------------------------------------------------
# Split enqueue helper
# ---------------------------------------------------------------------------


def _maybe_enqueue_split(
    subject: dict,
    root: Path,
    cfg: dict,
    split_depth: int = 0,
) -> bool:
    """
    Enqueue a split_subject action if the subject's message_count is at or
    above the configured threshold and no split action is already pending.
    Returns True if enqueued, False otherwise.
    """
    threshold = int(cfg.get("split_subject_threshold", 150))
    if threshold <= 0:
        return False
    if subject.get("state") in ("archived", "split"):
        return False
    count = int(subject.get("message_count") or 0)
    if count < threshold:
        return False
    subject_id = subject["id"]
    if _db.has_pending_memory_action("split_subject", subject_id, root):
        return False
    try:
        _db.enqueue_memory_action(
            action_type="split_subject",
            subject_id=subject_id,
            payload={"subject_name": subject["name"], "_split_depth": split_depth},
            actor=_db.ACTOR_MEMORY_ACTIONS,
            root=root,
        )
        _log.info(
            "SPLIT queued subject=%r id=%s (count=%d depth=%d)",
            subject["name"], subject_id[:8], count, split_depth,
        )
        return True
    except Exception as exc:
        _log.warning("SPLIT enqueue failed for subject=%r: %s", subject["name"], exc)
        return False


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_split_subject(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Partition an oversized subject into 2-5 child subjects.

    Asks the meta-model to divide the subject's memory into named children,
    creates each child in the Dewey tree with its own memory view, then sets
    the parent state to 'split' so daemon_reclass can re-attribute messages
    from the old parent to the appropriate children.
    """
    if not subject_id:
        raise ValueError("split_subject action missing subject_id")

    all_subjects = _subjects.load_subjects(root)
    subject = next((s for s in all_subjects if s["id"] == subject_id), None)
    if subject is None:
        raise LookupError(f"subject id={subject_id[:8]} not found")

    subject_name = str(payload.get("subject_name") or subject.get("name") or "")
    split_depth = int(payload.get("_split_depth", 0))
    _log.info("SPLIT starting subject=%r id=%s depth=%d", subject_name, subject_id[:8], split_depth)

    existing_content = _subjects.read_subject_memory(subject, root)
    existing_view = _subjects.parse_memory_view(existing_content)
    event_log = _subjects.parse_memory_log(existing_content)

    children_spec = meta_agent.split_subject_memory(
        subject_name=subject_name,
        existing_view=existing_view,
        event_log=event_log,
        cfg=cfg,
    )
    if not children_spec:
        raise RuntimeError(
            f"split_subject_memory returned no children for subject={subject_name!r}"
        )

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    child_subjects: list[dict] = []
    for child_spec in children_spec:
        child_name = child_spec["name"]
        child_view = child_spec["view"]
        try:
            child = _subjects.create_subject(
                name=child_name,
                description=f"Split from '{subject_name}'.",
                parent_id=subject_id,
                root=root,
            )
            child_content = _subjects.compose_memory_file(
                subject_name=child_name,
                view=child_view,
                new_log_entry=f"Created by splitting '{subject_name}'.",
                status="established",
                prior_log="",
                timestamp=timestamp,
            )
            _subjects.write_subject_memory(child, child_content, root)
            child_subjects.append(child)
            _log.info("SPLIT child created: %r id=%s", child_name, child["id"][:8])
        except Exception as exc:
            _log.warning("SPLIT could not create child %r: %s", child_name, exc)

    if not child_subjects:
        raise RuntimeError(f"no children could be created for subject={subject_name!r}")

    # Mark the parent as 'split' - daemon_reclass scans for this state and
    # re-attributes messages to the appropriate children before archiving.
    try:
        _subjects.set_subject_state(subject_id, "split", root)
        _log.info("SPLIT parent marked split: %r id=%s", subject_name, subject_id[:8])
    except Exception as exc:
        _log.warning("SPLIT could not mark parent as split %r: %s", subject_name, exc)

    # Depth-guarded recursive check: see if any child is itself over threshold.
    if split_depth < _SPLIT_MAX_DEPTH:
        for child in child_subjects:
            _maybe_enqueue_split(child, root, cfg, split_depth=split_depth + 1)

    _log.info(
        "SPLIT complete: subject=%r -> %d children",
        subject_name, len(child_subjects),
    )


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "split_subject": _handle_split_subject,
}

# ---------------------------------------------------------------------------
# Watch spec scanner (called by daemon_watcher on a schedule)
# ---------------------------------------------------------------------------


def _scan_oversized_subjects(root: Path) -> list[dict]:
    """
    Return subjects eligible for splitting that do not yet have a pending
    split_subject action.  Threshold is read from the live config so a
    config change takes effect on the next scan interval without restarting
    the scanner thread.

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    The watcher infrastructure calls has_pending_memory_action before enqueuing,
    so this scanner intentionally returns all over-threshold subjects - dedup
    is handled upstream.
    """
    cfg = _config.load()
    threshold = int(cfg.get("split_subject_threshold", 150))
    if threshold <= 0:
        return []
    try:
        subjects = _db.get_subjects_needing_split(threshold, root)
    except Exception as exc:
        _log.warning("SPLIT scan: DB query failed: %s", exc)
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
        "interval_s": 120,
        "action_type": "split_subject",
        "actor": _db.ACTOR_MEMORY_ACTIONS,
        "scanner": _scan_oversized_subjects,
    }
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="split",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
