"""
daemon_escalate.py - HuBrIS escalation daemon.

Single job: drain escalate_message actions from the memory_actions queue.

  escalate_message - run the heavy model over a list of null-assigned message
                     IDs that daemon_classify could not attribute to any
                     subject. Payload shape:
                       {"session_id": str, "message_ids": list[str]}

Run with: python -m daemon_escalate
"""

import datetime
from pathlib import Path
from typing import Any

import config as _config
import db as _db
import meta_agent
import subjects as _subjects
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.escalate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Token budget ceiling for a single escalation prompt.
_ESCALATION_TOKEN_MAX = 6000         # ~24 KB
# Minimum null-pool token count to justify calling the heavy model.
_ESCALATION_TOKEN_THRESHOLD = 300    # ~1.2 KB

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _save_assignments(
    root: Path,
    session_id: str,
    assignments: dict[int, Any],
) -> None:
    try:
        _db.save_assignments(session_id, assignments, _db.ACTOR_MESSAGE_CLASSIFIER, root)
    except Exception as exc:
        _log.warning("Could not persist assignments for %s: %s", session_id[:8], exc)


def _write_escalated_memory(
    root: Path,
    cfg: dict,
    session_id: str,
    subject_message_buffer: dict[str, list[dict]],
) -> None:
    relevance_floor = float(cfg.get("relevance_floor", 0.2))
    all_subjects = _subjects.load_subjects(root)
    subject_lookup = {s["id"]: s for s in all_subjects}
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    for subj_id, msgs in subject_message_buffer.items():
        subject = subject_lookup.get(subj_id)
        if subject is None:
            continue
        filtered = [m for m in msgs if float(m.get("relevance", 1.0)) >= relevance_floor]
        if not filtered:
            continue
        try:
            existing = _subjects.read_subject_memory(subject, root)
            view, log_entry, status = meta_agent.update_subject_memory(
                subject_name=subject["name"],
                existing_view=_subjects.parse_memory_view(existing),
                new_messages=filtered,
                cfg=cfg,
            )
            if view:
                content = _subjects.compose_memory_file(
                    subject_name=subject["name"],
                    view=view,
                    new_log_entry=log_entry,
                    status=status,
                    prior_log=_subjects.parse_memory_log(existing),
                    timestamp=timestamp,
                )
                _subjects.write_subject_memory(subject, content, root)
                _log.info(
                    "ESCALATE MEMORY WRITE %s: subject=%r %d msg(s) -> %d chars",
                    session_id[:8], subject["name"], len(filtered), len(content),
                )
        except Exception as exc:
            _log.warning(
                "ESCALATE MEMORY WRITE %s: subject=%r failed (%s)",
                session_id[:8], subject.get("name", subj_id), type(exc).__name__,
            )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_escalate_message(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Run the heavy model over a list of null-assigned message IDs.

    daemon_classify enqueues these when Stage 1 classification leaves messages
    without a subject assignment. This handler fetches their bodies, applies a
    token-budget check, and runs meta_agent.escalate_unassigned to assign them
    to existing or newly created subjects. Memory files are updated for any
    subject that receives new messages.

    On EscalateFailed the handler re-raises so the queue retries the action.
    """
    session_id = str(payload["session_id"])
    message_ids: list[str] = list(payload.get("message_ids", []))

    if not message_ids:
        _log.info("ESCALATE %s: empty message_ids - nothing to do", session_id[:8])
        return

    esc_msgs = _db.get_messages_by_ids(message_ids, root)
    if not esc_msgs:
        _log.warning(
            "ESCALATE %s: no messages found for %d IDs - abandoned",
            session_id[:8], len(message_ids),
        )
        return

    esc_token_max = int(cfg.get("escalation_token_max", _ESCALATION_TOKEN_MAX))
    esc_threshold = int(cfg.get("escalation_token_threshold", _ESCALATION_TOKEN_THRESHOLD))

    # Token-budget check - skip if the pool is too small for the heavy model.
    total_tokens = sum(meta_agent.estimate_tokens(m["content"]) for m in esc_msgs)
    if total_tokens < esc_threshold:
        _log.info(
            "ESCALATE DEFER %s: ~%d tokens across %d msg(s) (threshold %d) - skipping",
            session_id[:8], total_tokens, len(esc_msgs), esc_threshold,
        )
        return

    # Trim to per-prompt token ceiling.
    esc_msgs_trimmed: list[dict] = []
    running = 0
    for msg in esc_msgs:
        t = meta_agent.estimate_tokens(msg["content"])
        if running + t > esc_token_max and esc_msgs_trimmed:
            break
        esc_msgs_trimmed.append(msg)
        running += t

    _log.info(
        "ESCALATE %s: ~%d tokens across %d msg(s) - invoking meta-model",
        session_id[:8], running, len(esc_msgs_trimmed),
    )

    known = _subjects.load_subjects(root)
    valid_ids = {s["id"] for s in known}
    subject_message_buffer: dict[str, list[dict]] = {}
    relevance_by_index: dict[int, dict[str, float]] = {}

    try:
        escalation = meta_agent.escalate_unassigned(esc_msgs_trimmed, known, cfg)
    except meta_agent.EscalateFailed as exc:
        _log.warning("ESCALATE_FAILED %s: %s", session_id[:8], exc)
        raise

    for esc_local, subj_id in escalation["assignments"].items():
        if esc_local >= len(esc_msgs_trimmed):
            _log.warning(
                "ESCALATE %s: model returned out-of-bounds index %d - skipped",
                session_id[:8], esc_local,
            )
            continue
        msg = esc_msgs_trimmed[esc_local]
        if subj_id is None:
            continue
        if subj_id not in valid_ids:
            _log.warning(
                "ESCALATE %s: hallucinated subject id=%r - leaving null",
                session_id[:8], subj_id,
            )
            continue
        relevance_by_index[int(msg["message_index"])] = {subj_id: 1.0}
        subject_message_buffer.setdefault(subj_id, []).append({**msg, "relevance": 1.0})
        try:
            _subjects.increment_message_count(subj_id, delta=1, root=root)
        except Exception:
            pass

    for new_subj_def in escalation["new_subjects"]:
        try:
            raw_parent = new_subj_def.get("parent_id")
            safe_parent = raw_parent if raw_parent in valid_ids else None
            if raw_parent and not safe_parent:
                _log.warning(
                    "ESCALATE %s: hallucinated parent_id=%r - creating at root",
                    session_id[:8], raw_parent,
                )
            subject = _subjects.create_subject(
                name=new_subj_def["name"],
                description=new_subj_def.get("description", ""),
                parent_id=safe_parent,
                root=root,
            )
            new_id = subject["id"]
            valid_ids.add(new_id)
            for esc_msg_idx in new_subj_def.get("message_indices", []):
                if esc_msg_idx >= len(esc_msgs_trimmed):
                    _log.warning(
                        "ESCALATE %s: out-of-bounds message_index %d in new subject %r - skipped",
                        session_id[:8], esc_msg_idx, new_subj_def.get("name"),
                    )
                    continue
                msg = esc_msgs_trimmed[esc_msg_idx]
                relevance_by_index[int(msg["message_index"])] = {new_id: 1.0}
                subject_message_buffer.setdefault(new_id, []).append({**msg, "relevance": 1.0})
                try:
                    _subjects.increment_message_count(new_id, delta=1, root=root)
                except Exception:
                    pass
            _log.info(
                "ESCALATE CREATE %s: subject=%r id=%s",
                session_id[:8], subject["name"], new_id,
            )
        except Exception as exc:
            _log.warning(
                "ESCALATE CREATE %s: failed for %r (%s)",
                session_id[:8], new_subj_def.get("name"), type(exc).__name__,
            )

    if relevance_by_index:
        _save_assignments(root, session_id, relevance_by_index)

    if subject_message_buffer:
        _write_escalated_memory(root, cfg, session_id, subject_message_buffer)


# ---------------------------------------------------------------------------
# Action handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "escalate_message": _handle_escalate_message,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="escalate",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
