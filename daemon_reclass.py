"""
daemon_reclass.py - HuBrIS message re-classification daemon.

Single job: drain classify_memory actions from the memory_actions queue.

  classify_memory - re-attribute messages linked to a split parent subject
                    to the most appropriate child subject. Payload shape:
                      {"candidate_subject_ids": list[str]}

Run with: python -m daemon_reclass
"""

from pathlib import Path
from typing import Any

import config as _config
import db as _db
import meta_agent
import subjects as _subjects
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.reclass")


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _scan_split_subjects(root: Path) -> list[dict]:
    """
    Return subjects currently in the 'split' state so the watcher can enqueue
    classify_memory actions for each.  Subjects with no direct children are
    skipped with a warning - there is nothing to re-attribute them to.

    Each returned item has: subject_id, payload.candidate_subject_ids.
    """
    try:
        split = _db.get_split_subjects(root)
    except Exception as exc:
        _log.warning("RECLASS scan: DB query failed: %s", exc)
        return []
    items = []
    for parent in split:
        try:
            children = _db.get_children_of_subject(parent["id"], root)
        except Exception as exc:
            _log.warning(
                "RECLASS scan: could not load children for id=%s: %s",
                parent["id"][:8], exc,
            )
            continue
        if not children:
            _log.warning(
                "RECLASS scan: split subject id=%s has no children - skipping",
                parent["id"][:8],
            )
            continue
        items.append({
            "subject_id": parent["id"],
            "payload": {"candidate_subject_ids": [c["id"] for c in children]},
        })
    return items


_WATCH_SPEC: list[dict] = [
    {
        "interval_s": 60,
        "action_type": "classify_memory",
        "actor": _db.ACTOR_MEMORY_ACTIONS,
        "scanner": _scan_split_subjects,
    }
]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_classify_memory(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Re-attribute messages linked to a split parent to its child subjects.

    1. Load all messages linked to the split parent.
    2. Load the candidate child subjects from the payload.
    3. Classify each message against the children.
    4. Delete the old links and write new ones to the winning child.
       Unmatched messages fall back to the first child.
    5. Set the parent state to 'archived'.
    """
    if not subject_id:
        raise ValueError("classify_memory action missing subject_id")

    # Load messages linked to the split parent.
    messages = _db.get_messages_linked_to_subject(subject_id, root)
    if not messages:
        _log.info(
            "RECLASS subject_id=%s: no linked messages - archiving immediately",
            subject_id[:8],
        )
        _subjects.set_subject_state(subject_id, "archived", root)
        return

    # Load candidate children from the payload.
    candidate_ids: list[str] = payload.get("candidate_subject_ids") or []
    children = [_db.get_subject(cid, root) for cid in candidate_ids]
    children = [c for c in children if c is not None]
    if not children:
        raise ValueError(
            f"classify_memory: no valid children for subject_id={subject_id}"
        )

    # Classify messages against children.
    result = meta_agent.classify_messages(messages, children, cfg)

    # Build assignments: message_id -> winning child_id.
    # Fall back to the first child for messages the classifier left unmatched.
    fallback_id = children[0]["id"]
    assignments: dict[str, str] = {}
    for i, msg in enumerate(messages):
        scores = result.get(i) or {}
        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else fallback_id
        assignments[msg["id"]] = best

    # Delete old links, write new ones.
    _db.reattribute_links(subject_id, assignments, _db.ACTOR_MEMORY_ACTIONS, root)
    _log.info(
        "RECLASS subject_id=%s: re-attributed %d message(s) across %d child(ren)",
        subject_id[:8], len(assignments), len(children),
    )

    # Archive the split parent - its messages are now re-attributed.
    _subjects.set_subject_state(subject_id, "archived", root)
    _log.info("RECLASS parent archived: id=%s", subject_id[:8])


# ---------------------------------------------------------------------------
# Action handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "classify_memory": _handle_classify_memory,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="reclass",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
