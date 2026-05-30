"""
daemon_embed.py - HuBrIS message and subject embedding daemon.

Single job: drain embed_messages and embed_subjects actions from the
memory_actions queue.

  embed_messages - embed a batch of un-vectorized messages for one session.
                   Payload shape: {"session_id": str}

  embed_subjects - embed a batch of subjects whose memory_content has been
                   updated since their last embedding (or has never been
                   embedded). Payload shape: {}

Also exports _WATCH_SPEC so daemon_watcher discovers this daemon's scanners at
startup and enqueues embedding work on a schedule without daemon_watcher needing
any hardcoded knowledge of this daemon.

Silently does nothing when sqlite-vec is not installed or embed_model is not
configured - same guard pattern as daemon_reconcile.

Run with: python daemon_embed.py
"""

from pathlib import Path
from typing import Any

import config as _config
import db as _db
import embeddings as _emb
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.embed")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MESSAGES_BATCH = 20
_SUBJECTS_BATCH = 20


# ---------------------------------------------------------------------------
# Watch spec scanners (called by daemon_watcher on a schedule)
# ---------------------------------------------------------------------------


def _scan_messages_needing_embedding(root: Path) -> list[dict]:
    """
    Return embed_messages items for sessions whose total message count exceeds
    the vectorized_count HWM.

    Each item shape:
      subject_id - session_id, used as the dedup key in memory_actions
      payload    - {"session_id": str}

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    Returns an empty list when sqlite-vec is not installed.
    """
    if not _emb.is_available():
        return []
    try:
        vectorized = _db.load_vectorized_counts(root)
        totals = _db.get_session_message_counts(root)
    except Exception as exc:
        _log.warning("EMBED scan (messages): DB query failed: %s", exc)
        return []

    items = []
    for session_id, total in totals.items():
        hwm = vectorized.get(session_id, 0)
        if total > hwm:
            items.append({
                "subject_id": session_id,
                "payload": {"session_id": session_id},
            })
    return items


def _scan_subjects_needing_embedding(root: Path) -> list[dict]:
    """
    Return a single embed_subjects item if any subjects need (re-)embedding.

    Uses subject_id=None so at most one embed_subjects action is pending at a
    time - the handler processes a batch internally on each drain cycle.

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    Returns an empty list when sqlite-vec is not installed.
    """
    if not _emb.is_available():
        return []
    try:
        rows = _db.get_subjects_needing_embedding(1, root)
    except Exception as exc:
        _log.warning("EMBED scan (subjects): DB query failed: %s", exc)
        return []

    if not rows:
        return []
    return [{"subject_id": None, "payload": {}}]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_embed_messages(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Embed up to _MESSAGES_BATCH un-vectorized messages for one session.

    Reads the vectorized_count HWM fresh from DB so stale payloads never cause
    double-embedding. Advances the HWM after each successful batch.
    Stops on the first embed failure so the next cycle retries from there.
    """
    if not _emb.is_available():
        return
    model = cfg.get("embed_model", "").strip()
    if not model:
        return

    session_id: str = payload["session_id"]
    vectorized = _db.load_vectorized_counts(root)
    from_index = vectorized.get(session_id, 0)

    rows = _db.get_messages_needing_embedding(session_id, from_index, _MESSAGES_BATCH, root)
    if not rows:
        return

    embeddings_by_rowid: dict[int, bytes] = {}
    highest_index = from_index
    for row in rows:
        text = str(row.get("raw_content") or "").strip()
        if not text:
            # Advance past empty/null messages without embedding them.
            highest_index = max(highest_index, int(row["message_index"]) + 1)
            continue
        emb_bytes = _emb.embed(text, cfg)
        if emb_bytes is None:
            # Embedding failed (model not loaded, Ollama down, etc.).
            # Stop here so we retry this message on the next cycle.
            break
        embeddings_by_rowid[int(row["rowid"])] = emb_bytes
        highest_index = max(highest_index, int(row["message_index"]) + 1)

    if embeddings_by_rowid:
        _db.upsert_message_embeddings_batch(embeddings_by_rowid, root)
        _log.debug(
            "EMBED MESSAGES %s: embedded %d message(s) (HWM %d -> %d)",
            session_id[:8], len(embeddings_by_rowid), from_index, highest_index,
        )

    if highest_index > from_index:
        vectorized[session_id] = highest_index
        _db.save_vectorized_counts(vectorized, _db.ACTOR_EMBEDDING_WRITER, root)


def _handle_embed_subjects(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Embed up to _SUBJECTS_BATCH subjects whose memory_content has been updated
    since their last embedding (or has never been embedded).

    Stops on the first embed failure so the next cycle retries from there.
    """
    if not _emb.is_available():
        return
    model = cfg.get("embed_model", "").strip()
    if not model:
        return

    rows = _db.get_subjects_needing_embedding(_SUBJECTS_BATCH, root)
    if not rows:
        return

    embedded = 0
    for row in rows:
        text = str(row.get("memory_content") or "").strip()
        if not text:
            continue
        emb_bytes = _emb.embed(text, cfg)
        if emb_bytes is None:
            # Embedding failed - stop and retry next cycle.
            break
        _db.upsert_subject_embedding(row["id"], emb_bytes, root)
        embedded += 1

    if embedded:
        _log.debug("EMBED SUBJECTS: embedded %d subject(s)", embedded)


# ---------------------------------------------------------------------------
# Watch spec (read by daemon_watcher at startup via _WATCH_SPEC discovery)
# ---------------------------------------------------------------------------

_WATCH_SPEC: list[dict[str, Any]] = [
    {
        "interval_s": 30,
        "action_type": "embed_messages",
        "actor": _db.ACTOR_EMBEDDING_WRITER,
        "scanner": _scan_messages_needing_embedding,
    },
    {
        "interval_s": 30,
        "action_type": "embed_subjects",
        "actor": _db.ACTOR_EMBEDDING_WRITER,
        "scanner": _scan_subjects_needing_embedding,
    },
]


# ---------------------------------------------------------------------------
# Action handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "embed_messages": _handle_embed_messages,
    "embed_subjects": _handle_embed_subjects,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="embed",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
