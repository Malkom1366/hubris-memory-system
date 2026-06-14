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
    Return embed_messages items for sessions that have at least one non-empty
    message not yet present in message_chunks.

    Each item shape:
      subject_id - session_id, used as the dedup key in memory_actions
      payload    - {"session_id": str}

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    Returns an empty list when sqlite-vec is not installed.
    """
    if not _emb.is_available():
        return []
    try:
        sessions = _db.get_sessions_needing_chunking(root)
    except Exception as exc:
        _log.warning("EMBED scan (messages): DB query failed: %s", exc)
        return []

    return [
        {"subject_id": session_id, "payload": {"session_id": session_id}}
        for session_id in sessions
    ]


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
    Chunk and embed up to _MESSAGES_BATCH un-chunked messages for one session.

    Each non-empty message is split into overlapping character chunks and each
    chunk gets its own embedding row in vec_message_chunks (keyed through
    message_chunks). Short messages produce a single chunk; long messages
    produce multiple, so every part of the message is searchable.

    Stops on the first embed failure so the next cycle retries from that
    message onward.
    """
    if not _emb.is_available():
        return
    model = cfg.get("embed_model", "").strip()
    if not model:
        return

    session_id: str = payload["session_id"]
    rows = _db.get_messages_needing_chunking(session_id, _MESSAGES_BATCH, root)
    if not rows:
        return

    embedded_messages = 0
    total_chunks = 0
    for row in rows:
        text = str(row.get("raw_content") or "").strip()
        if not text:
            continue  # scan excludes empty messages; belt-and-suspenders guard
        chunk_embeddings = _emb.embed_chunks(text, cfg)
        if not chunk_embeddings:
            # Embedding failed - stop here so we retry this message next cycle.
            break
        written = _db.upsert_message_chunk_embeddings(str(row["id"]), chunk_embeddings, root)
        if written:
            embedded_messages += 1
            total_chunks += written

    if embedded_messages:
        _log.debug(
            "EMBED MESSAGES %s: %d message(s) -> %d chunk(s) written",
            session_id[:8], embedded_messages, total_chunks,
        )


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
