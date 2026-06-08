"""
daemon_compact.py - HuBrIS proactive context compaction daemon.

Single job: drain compact_memory actions from the memory_actions queue.

  compact_memory - archives cold subjects and writes tombstones via the
                   frontend adapter, freeing context window space before
                   the model hits its auto-compaction limit or FIFO context
                   begins dropping old messages.

Run with: python -m daemon_compact [--adapter:<name>]
"""

import sys
from pathlib import Path
from typing import Any

import config as _config
import db as _db
import subjects as _subjects
from daemon_base import DaemonBase
from frontend_adapters import SessionAdapter, build_active_frontend_adapters
from log import get_logger

_log = get_logger("hubris.compact")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Proactive compaction threshold (tokens).  Overridden by payload["threshold"]
# when the action was enqueued with an explicit value.
_CONTEXT_COMPACT_THRESHOLD = 80_000  # ~320 KB

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_adapter: SessionAdapter | None = None

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


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
    "compact_memory": _handle_compact_memory,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()

    root = _config.memory_root(cfg.get("workspace_id", "global"))

    global _adapter
    adapters = build_active_frontend_adapters(cfg)
    if adapters:
        _adapter = adapters[0]
        _log.info("COMPACT DAEMON: adapter=%s", type(_adapter).__name__)
    else:
        _log.warning("COMPACT DAEMON: no frontend adapter available")

    daemon = DaemonBase(
        name="compact",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_MEMORY_ACTIONS,
    )
    daemon.run()


if __name__ == "__main__":
    main()
