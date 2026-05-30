"""
One-shot reset helper for the SQLite-backed storage layer.

Blacklists the configured sessions, resets subject message counts, and clears
session progress so the next run starts from a known state.
"""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as _config
import db as _db

ROOT = _config.memory_root("global")

SESSIONS_TO_BLACKLIST = [
    "f261dd8f-a7b5-4633-a7fc-0c2b808c0f30",
    "a09334f8-6198-49c6-aacd-68ad8755fe76",
    "4e47c765-2910-4159-aee3-3a5853ebc7df",
    "d20e4f67-15eb-409c-8cfb-5a0b6499d7ef",
    "e1329148-cfc0-44bf-b944-3ffc45c68b6a",
]

_db.init_db(ROOT)
_db.save_session_blacklist(set(SESSIONS_TO_BLACKLIST), _db.ACTOR_BLACKLIST_MANAGER, ROOT)
print(f"Blacklisted {len(SESSIONS_TO_BLACKLIST)} sessions -> {_db.db_path(ROOT)}")

with _db.connect(ROOT) as conn:
    subject_count = conn.execute("SELECT COUNT(*) AS c FROM semantic_subjects").fetchone()["c"]
    conn.execute(
        """
        UPDATE semantic_subjects
        SET message_count = 0,
            updated_date_utc = CURRENT_TIMESTAMP,
            updated_by = ?
        """,
        (_db.ACTOR_SUBJECT_STATE_MANAGER,),
    )
    conn.execute("DELETE FROM session_progress")
    conn.execute("DELETE FROM classify_failures")
    remaining_progress = conn.execute("SELECT COUNT(*) AS c FROM session_progress").fetchone()["c"]

print(f"Reset message_count on {subject_count} subject(s) -> {_db.db_path(ROOT)}")
print(f"session_progress rows remaining: {remaining_progress}")
