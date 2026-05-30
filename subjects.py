"""
Subject management for MCMCP.

A "subject" is a named topic within a session. Subjects are the nodes of the
Dewey-style memory tree and are persisted in the workspace-scoped SQLite store.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import catalog
import config as _config
import db as _db
from log import get_logger

_log = get_logger("hubris.subjects")

_SUBJECTS_DIR = "subjects"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subjects_dir(root: Path) -> Path:
    d = root / _SUBJECTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Manifest load / save
# ---------------------------------------------------------------------------

def load_subjects(root: Path | None = None) -> list[dict[str, Any]]:
    """Return the subjects manifest as a list of dicts."""
    if root is None:
        root = _config.memory_root()
    return _db.load_subjects(root)


def save_subjects(subjects: list[dict[str, Any]], root: Path | None = None) -> None:
    """Persist the subjects manifest."""
    if root is None:
        root = _config.memory_root()
    _db.save_subjects(subjects, root)


# ---------------------------------------------------------------------------
# Dewey ID assignment
# ---------------------------------------------------------------------------

def _top_level_ids(subjects: list[dict]) -> list[str]:
    return [s["dewey_id"] for s in subjects if "." not in s["dewey_id"]]


def _child_ids(subjects: list[dict], parent_id: str) -> list[str]:
    prefix = parent_id + "."
    return [
        s["dewey_id"] for s in subjects
        if s["dewey_id"].startswith(prefix)
        and s["dewey_id"].count(".") == parent_id.count(".") + 1
    ]


def assign_dewey_id(subjects: list[dict], parent_id: str | None = None) -> str:
    """
    Return the next available Dewey ID.
    If parent_id is None, assigns a top-level ID (0, 1, 2 ...).
    If parent_id is given, assigns a child ID (parent.0, parent.1 ...).
    """
    if parent_id is None:
        existing = _top_level_ids(subjects)
        next_int = len(existing)
        return str(next_int)
    else:
        existing = _child_ids(subjects, parent_id)
        next_int = len(existing)
        return f"{parent_id}.{next_int}"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_subject(
    name: str,
    description: str = "",
    parent_id: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Create and persist a new subject. Returns the new subject dict.
    Raises ValueError if a subject with the same name already exists.
    """
    if root is None:
        root = _config.memory_root()
    subjects = load_subjects(root)
    if any(s["name"].lower() == name.lower() for s in subjects):
        raise ValueError(f"Subject '{name}' already exists.")
    # Resolve parent_id (short hex or dewey_id) to the parent's dewey_id.
    parent_dewey_id: str | None = None
    if parent_id is not None:
        parent_subject = next(
            (s for s in subjects if s["id"] == parent_id or s["dewey_id"] == parent_id),
            None,
        )
        if parent_subject is None:
            raise KeyError(f"Parent subject '{parent_id}' not found.")
        parent_dewey_id = parent_subject["dewey_id"]
    dewey_id = assign_dewey_id(subjects, parent_dewey_id)
    subject: dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "dewey_id": dewey_id,
        "parent_id": parent_subject["id"] if parent_id is not None else None,
        "name": name,
        "description": description,
        "state": "open",
        "message_count": 0,
        "last_activity": _now_iso(),
        "created": _now_iso(),
        "created_by": _db.ACTOR_SUBJECT_GENERATOR,
        "updated_by": _db.ACTOR_SUBJECT_GENERATOR,
    }
    _db.create_subject_record(subject, root)
    return subject


def get_subject(subject_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """Look up a subject by its short id or dewey_id. Returns None if not found."""
    if root is None:
        root = _config.memory_root()
    return _db.get_subject(subject_id, root)


def set_subject_state(
    subject_id: str,
    state: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Transition a subject to the given state. Returns the updated subject."""
    if root is None:
        root = _config.memory_root()
    existing = get_subject(subject_id, root)
    if existing is None:
        raise KeyError(f"Subject '{subject_id}' not found.")
    last_activity = _now_iso()
    _db.update_subject_state(
        subject_id,
        state,
        last_activity,
        _db.ACTOR_SUBJECT_STATE_MANAGER,
        root,
    )
    updated = get_subject(subject_id, root)
    if updated is None:
        raise KeyError(f"Subject '{subject_id}' not found.")
    return updated


def move_subject(
    subject_id: str,
    new_parent_id: str | None,
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Move a subject to a new parent (or to top-level if new_parent_id is None).
    Updates the subject's Dewey ID and all its descendants' Dewey IDs.
    Returns a dict with old_dewey, new_dewey, subject id/name, and descendant count.
    """
    if root is None:
        root = _config.memory_root()
    subjects = load_subjects(root)

    subject = next(
        (s for s in subjects if s["id"] == subject_id or s["dewey_id"] == subject_id),
        None,
    )
    if subject is None:
        raise KeyError(f"Subject '{subject_id}' not found.")

    new_parent_dewey: str | None = None
    new_parent_subject_id: str | None = None
    if new_parent_id is not None:
        new_parent = next(
            (s for s in subjects if s["id"] == new_parent_id or s["dewey_id"] == new_parent_id),
            None,
        )
        if new_parent is None:
            raise KeyError(f"New parent '{new_parent_id}' not found.")
        if new_parent["id"] == subject["id"]:
            raise ValueError("A subject cannot be its own parent.")
        if new_parent["dewey_id"].startswith(subject["dewey_id"] + "."):
            raise ValueError(
                f"Cannot move '{subject['name']}' under one of its own descendants."
            )
        new_parent_dewey = new_parent["dewey_id"]
        new_parent_subject_id = new_parent["id"]

    old_dewey = subject["dewey_id"]
    # Exclude the subject itself from the subjects list when computing the new Dewey ID
    # so its current slot does not inflate the sibling count.
    subjects_without_self = [s for s in subjects if s["id"] != subject["id"]]
    new_dewey = assign_dewey_id(subjects_without_self, new_parent_dewey)

    # Collect descendant updates: any subject whose dewey_id starts with old_dewey + "."
    descendant_updates: list[tuple[str, str]] = []
    for s in subjects:
        if s["dewey_id"].startswith(old_dewey + "."):
            suffix = s["dewey_id"][len(old_dewey):]  # e.g. ".0" or ".1.2"
            descendant_updates.append((s["id"], new_dewey + suffix))

    _db.update_subject_parent_and_dewey(
        subject["id"],
        new_dewey,
        new_parent_subject_id,
        descendant_updates,
        _db.ACTOR_USER,
        root,
    )

    return {
        "id": subject["id"],
        "name": subject["name"],
        "old_dewey": old_dewey,
        "new_dewey": new_dewey,
        "old_parent_id": subject.get("parent_id"),
        "new_parent_id": new_parent_subject_id,
        "descendants_updated": len(descendant_updates),
    }


def increment_message_count(subject_id: str, delta: int = 1, root: Path | None = None) -> None:
    """Bump the message_count and last_activity for a subject."""
    if root is None:
        root = _config.memory_root()
    _db.increment_subject_message_count(
        subject_id,
        delta,
        _now_iso(),
        _db.ACTOR_MESSAGE_CLASSIFIER,
        root,
    )


# ---------------------------------------------------------------------------
# Memory file access
# ---------------------------------------------------------------------------

def subject_memory_path(subject: dict[str, Any], root: Path | None = None) -> Path:
    """Return the legacy conceptual memory path for a subject."""
    if root is None:
        root = _config.memory_root()
    safe_id = subject["dewey_id"].replace(".", "-")
    return _subjects_dir(root) / f"{safe_id}.md"


def write_subject_memory(
    subject: dict[str, Any],
    content: str,
    root: Path | None = None,
) -> None:
    """Write (overwrite) the subject memory content in SQLite."""
    if root is None:
        root = _config.memory_root()
    _db.write_subject_memory(subject["id"], content, _db.ACTOR_MEMORY_WRITER, root)


def read_subject_memory(
    subject: dict[str, Any],
    root: Path | None = None,
) -> str:
    """Read the subject memory content. Returns '' if no memory exists yet."""
    if root is None:
        root = _config.memory_root()
    return _db.read_subject_memory(subject["id"], root)


# ---------------------------------------------------------------------------
# Memory file format helpers
# ---------------------------------------------------------------------------

_VIEW_START = "<!-- hubris:view -->"
_VIEW_END = "<!-- /hubris:view -->"
_LOG_START = "<!-- hubris:log -->"
_LOG_END = "<!-- /hubris:log -->"
_LEGACY_VIEW_START = "<!-- mcmcp:view -->"
_LEGACY_VIEW_END = "<!-- /mcmcp:view -->"
_LEGACY_LOG_START = "<!-- mcmcp:log -->"
_LEGACY_LOG_END = "<!-- /mcmcp:log -->"


def parse_memory_view(content: str) -> str:
    """
    Extract the materialized view section from a memory file.
    Returns the view text stripped of its delimiters, or '' if absent.
    Legacy files that predate the structured format return '' so callers
    can fall back to the raw file content.
    """
    start = content.find(_VIEW_START)
    end = content.find(_VIEW_END)
    if start == -1 or end == -1:
        # Try legacy markers for files written before the hubris rename.
        start = content.find(_LEGACY_VIEW_START)
        end = content.find(_LEGACY_VIEW_END)
        if start == -1 or end == -1:
            return ""
        return content[start + len(_LEGACY_VIEW_START):end].strip()
    return content[start + len(_VIEW_START):end].strip()


def parse_memory_log(content: str) -> str:
    """
    Extract the event log section from a memory file.
    Returns the log text stripped of its delimiters, or '' if absent.
    """
    start = content.find(_LOG_START)
    end = content.find(_LOG_END)
    if start == -1 or end == -1:
        # Try legacy markers for files written before the hubris rename.
        start = content.find(_LEGACY_LOG_START)
        end = content.find(_LEGACY_LOG_END)
        if start == -1 or end == -1:
            return ""
        return content[start + len(_LEGACY_LOG_START):end].strip()
    return content[start + len(_LOG_START):end].strip()


def compose_memory_file(
    subject_name: str,
    view: str,
    new_log_entry: str,
    status: str,
    prior_log: str,
    timestamp: str,
) -> str:
    """
    Assemble a full memory file from a materialized view, a new event log entry,
    the prior log content (from parse_memory_log), and an ISO timestamp.

    File structure:
      # Subject Name
      *Last updated: <timestamp>*

      <!-- hubris:view -->
      <materialized view - what recall() returns>
      <!-- /hubris:view -->

      <!-- hubris:log -->
      ### <timestamp> | <status>
      <log_entry>
      [prior entries appended below]
      <!-- /hubris:log -->
    """
    log_parts: list[str] = []
    log_parts.append(f"### {timestamp} | {status}\n{new_log_entry}")
    if prior_log:
        log_parts.append(prior_log)
    log_content = "\n\n".join(log_parts)

    return (
        f"# {subject_name}\n\n"
        f"*Last updated: {timestamp}*\n\n"
        f"{_VIEW_START}\n{view}\n{_VIEW_END}\n\n"
        f"{_LOG_START}\n{log_content}\n{_LOG_END}\n"
    )


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_subjects_summary(root: Path | None = None) -> list[dict[str, Any]]:
    """
    Return a summary list suitable for display: id, dewey_id, name, state,
    message_count, last_activity. Does not include full memory content.
    """
    if root is None:
        root = _config.memory_root()
    subjects = load_subjects(root)
    return [
        {
            "id": s["id"],
            "dewey_id": s["dewey_id"],
            "name": s["name"],
            "description": s.get("description", ""),
            "state": s["state"],
            "message_count": s.get("message_count", 0),
            "last_activity": s.get("last_activity", ""),
        }
        for s in sorted(subjects, key=lambda x: x["dewey_id"])
    ]


# ---------------------------------------------------------------------------
# Subject archiving
# ---------------------------------------------------------------------------

def archive_subject(subject: dict, root: Path) -> tuple[int, int]:
    """
    Mark `subject` as archived and enqueue it for Phase 2 finalization.

    No session-file rewrite. No autobiographical-memory mutation. Existing
    semantic_links from this subject to messages are kept in place so that
    history queries (recall, semantic search) still attribute those turns
    correctly. The subject's accumulated memory_content is preserved and
    will be reachable via recall(subject_name).

    Returns (0, 0) - preserved for backward compatibility with old call sites.
    """
    subject_id = subject["id"]
    subject_name = subject["name"]

    try:
        set_subject_state(subject_id, "archived", root)
        try:
            _db.enqueue_memory_action(
                action_type="finalize_subject",
                subject_id=subject_id,
                payload={"subject_name": subject_name},
                actor=_db.ACTOR_MEMORY_ACTIONS,
                root=root,
            )
            _log.info("FINALIZE queued subject=%r id=%s", subject_name, subject_id[:8])
        except Exception as exc:
            _log.warning("Could not enqueue finalize action: %s", exc)
    except (KeyError, ValueError) as exc:
        _log.warning("ARCHIVE: could not set subject state: %s", exc)

    _log.info(
        "SUBJECT ARCHIVED id=%s name=%r (status-only flip; history retained)",
        subject_id, subject_name,
    )
    return 0, 0


def run_lifecycle_pass(root: Path, cfg: dict, adapter: Any) -> tuple[int, int]:
    """
    Promote open subjects to dormant and dormant subjects to archived based on
    elapsed time since last_activity.  Rebuilds and broadcasts the catalog anchor
    if any state transitions occurred.

    Thresholds are read from cfg:
      dormant_after_minutes  (default 60)  - open -> dormant
      archive_after_minutes  (default 120) - dormant -> archived (total from last activity)

    Returns (promoted, archived) counts.  All errors are logged; none propagate.
    """
    dormant_after = timedelta(minutes=int(cfg.get("dormant_after_minutes", 60)))
    archive_after = timedelta(minutes=int(cfg.get("archive_after_minutes", 120)))
    now = datetime.now(timezone.utc)

    all_subjects = load_subjects(root)
    to_dormant: list[tuple[dict, timedelta]] = []
    to_archive: list[tuple[dict, timedelta]] = []

    for s in all_subjects:
        if s["state"] in ("archived", "split"):
            continue
        last_act_str = s.get("last_activity")
        if not last_act_str:
            continue
        try:
            age = now - datetime.fromisoformat(last_act_str)
        except ValueError:
            continue
        if s["state"] == "open" and age >= dormant_after:
            to_dormant.append((s, age))
        elif s["state"] == "dormant" and age >= archive_after:
            to_archive.append((s, age))

    promoted = 0
    for s, age in to_dormant:
        try:
            set_subject_state(s["id"], "dormant", root)
            promoted += 1
            _log.info(
                "AUTO-DORMANT subject=%r (inactive %.0f min)",
                s["name"], age.total_seconds() / 60,
            )
        except Exception as exc:
            _log.warning("AUTO-DORMANT subject=%r: %s", s.get("name"), exc)

    archived = 0
    for s, age in to_archive:
        try:
            _log.info(
                "AUTO-ARCHIVE subject=%r (inactive %.0f min)",
                s["name"], age.total_seconds() / 60,
            )
            archive_subject(s, root)
            archived += 1
        except Exception as exc:
            _log.warning("AUTO-ARCHIVE subject=%r: %s", s.get("name"), exc)

    if to_dormant or archived:
        try:
            rebuilt = catalog.rebuild_catalog_from_subjects(load_subjects(root), root)
            catalog.inject_anchor_all_sessions(adapter, rebuilt)
        except Exception as exc:
            _log.warning("AUTO-LIFECYCLE catalog broadcast failed: %s", exc)

    return promoted, archived
