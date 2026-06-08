from __future__ import annotations

"""
SQLite-backed persistence for HuBrIS.

This module replaces the flat-file manifest and JSON sidecars with a single
workspace-scoped database file under ~/.hubris/<workspace_id>/.
"""

import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import config as _config
from log import get_logger

_log = get_logger("hubris.db")

# Tracks DB paths whose stale 'running' memory_actions rows have been swept
# back to 'pending' during this process. The sweep is a single-process crash
# recovery, not something we want to run on every init_db() call (every helper
# in this module calls init_db). Without this guard, helpers like
# claim_next_memory_action would immediately re-pendingize their own claims.
_stale_running_swept: set[str] = set()

DB_FILENAME = "hubris.db"

ACTOR_MEMORY_WRITER = "MemoryWriter"
ACTOR_SUBJECT_GENERATOR = "SubjectGenerator"
ACTOR_SUBJECT_STATE_MANAGER = "SubjectStateManager"
ACTOR_MESSAGE_CLASSIFIER = "MessageClassifier"
ACTOR_SESSION_TRACKER = "SessionTracker"
ACTOR_BLACKLIST_MANAGER = "BlacklistManager"
ACTOR_WHITELIST_MANAGER = "WhitelistManager"
ACTOR_MEMORY_ACTIONS = "MemoryActions"
ACTOR_USER = "User"
ACTOR_CONFIDENCE_ATTENUATOR = "ConfidenceAttenuator"
ACTOR_EMBEDDING_WRITER = "EmbeddingWriter"

# Memory action lifecycle constants.
MEMORY_ACTION_STATUS_PENDING = "pending"
MEMORY_ACTION_STATUS_RUNNING = "running"
MEMORY_ACTION_STATUS_FAILED = "failed"
MEMORY_ACTION_MAX_ATTEMPTS = 5


def db_path(root: Path | None = None) -> Path:
    if root is None:
        root = _config.memory_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / DB_FILENAME


def connect(root: Path | None = None) -> sqlite3.Connection:
    path = db_path(root)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(root: Path | None = None) -> None:
    with connect(root) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT,
                first_message_utc TEXT,
                last_message_utc TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS autobiographical_memory (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                speaker TEXT NOT NULL CHECK(speaker IN ('user', 'assistant', 'system', 'tool')),
                raw_content TEXT NOT NULL,
                cleaned_content TEXT,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, message_index)
            );

            CREATE TABLE IF NOT EXISTS semantic_subjects (
                id TEXT PRIMARY KEY,
                dewey_id TEXT NOT NULL UNIQUE,
                parent_subject_id TEXT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                description TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                last_activity TEXT,
                memory_content TEXT NOT NULL DEFAULT '',
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                synthesized_at TEXT,
                FOREIGN KEY(parent_subject_id) REFERENCES semantic_subjects(id)
            );

            CREATE TABLE IF NOT EXISTS semantic_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                reconciled_at TEXT,
                FOREIGN KEY(message_id) REFERENCES autobiographical_memory(id) ON DELETE CASCADE,
                FOREIGN KEY(subject_id) REFERENCES semantic_subjects(id) ON DELETE CASCADE,
                UNIQUE(message_id, subject_id)
            );

            CREATE TABLE IF NOT EXISTS session_progress (
                session_id TEXT PRIMARY KEY,
                claimed_count INTEGER,
                committed_count INTEGER,
                vectorized_count INTEGER,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_blacklist (
                session_id TEXT PRIMARY KEY,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_whitelist (
                session_id TEXT PRIMARY KEY,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS message_blacklist (
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'blacklisted',
                reason TEXT NOT NULL DEFAULT 'unknown',
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL,
                PRIMARY KEY(session_id, message_index)
            );

            CREATE TABLE IF NOT EXISTS workspace_blacklist (
                workspace_id TEXT PRIMARY KEY,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS classify_failures (
                failure_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                subject_id TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'running', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                updated_by TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subject_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                from_link_id INTEGER NOT NULL,
                to_link_id INTEGER NOT NULL,
                relation TEXT NOT NULL CHECK(relation IN ('supports', 'contradicts', 'updates')),
                created_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT NOT NULL,
                FOREIGN KEY(subject_id) REFERENCES semantic_subjects(id) ON DELETE CASCADE,
                FOREIGN KEY(from_link_id) REFERENCES semantic_links(id) ON DELETE CASCADE,
                FOREIGN KEY(to_link_id) REFERENCES semantic_links(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS subject_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL UNIQUE,
                embedded_date_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(subject_id) REFERENCES semantic_subjects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL CHECK(operation IN ('INSERT', 'UPDATE', 'DELETE')),
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                old_values TEXT,
                new_values TEXT,
                changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                changed_by TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_session_index
                ON autobiographical_memory(session_id, message_index);

            CREATE INDEX IF NOT EXISTS idx_links_subject
                ON semantic_links(subject_id);

            CREATE INDEX IF NOT EXISTS idx_links_message
                ON semantic_links(message_id);

            CREATE INDEX IF NOT EXISTS idx_subjects_parent
                ON semantic_subjects(parent_subject_id);

            CREATE INDEX IF NOT EXISTS idx_memory_actions_pending
                ON memory_actions(status, action_type, id);

            CREATE INDEX IF NOT EXISTS idx_subject_relations_subject
                ON subject_relations(subject_id);

            CREATE INDEX IF NOT EXISTS idx_subject_embeddings_subject
                ON subject_embeddings(subject_id);
            """
        )
        # Schema migrations: add columns that were introduced after initial
        # table creation. SQLite does not support ADD COLUMN IF NOT EXISTS,
        # so we attempt each ALTER and silently ignore duplicate-column errors.
        _migrations = [
            "ALTER TABLE message_blacklist ADD COLUMN status TEXT NOT NULL DEFAULT 'blacklisted'",
            "ALTER TABLE message_blacklist ADD COLUMN reason TEXT NOT NULL DEFAULT 'unknown'",
        ]
        for _sql in _migrations:
            try:
                conn.execute(_sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        _create_audit_triggers(conn)
        # Single-process recovery: any rows left in 'running' from a prior
        # crashed server cycle get reset back to 'pending' so they re-run.
        # Only run once per DB path per process; otherwise every helper call
        # would clobber its own freshly-claimed row.
        path_key = str(db_path(root))
        if path_key not in _stale_running_swept:
            try:
                conn.execute(
                    "UPDATE memory_actions SET status = 'pending' WHERE status = 'running'"
                )
            except sqlite3.OperationalError:
                pass
            _stale_running_swept.add(path_key)
        # Optional: vec_messages virtual table (requires sqlite-vec to be installed).
        _maybe_create_vec_table(conn)


def _create_audit_triggers(conn: sqlite3.Connection) -> None:
    specs = [
        (
            "sessions",
            "NEW.id",
            "'workspace_id=' || COALESCE(NEW.workspace_id, '') || ';message_count=' || COALESCE(NEW.message_count, 0)",
            "'workspace_id=' || COALESCE(OLD.workspace_id, '') || ';message_count=' || COALESCE(OLD.message_count, 0)",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "autobiographical_memory",
            "NEW.id",
            "'session_id=' || NEW.session_id || ';message_index=' || NEW.message_index || ';speaker=' || NEW.speaker",
            "'session_id=' || OLD.session_id || ';message_index=' || OLD.message_index || ';speaker=' || OLD.speaker",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "semantic_subjects",
            "NEW.id",
            "'dewey_id=' || NEW.dewey_id || ';name=' || NEW.name || ';state=' || NEW.state || ';message_count=' || NEW.message_count",
            "'dewey_id=' || OLD.dewey_id || ';name=' || OLD.name || ';state=' || OLD.state || ';message_count=' || OLD.message_count",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "semantic_links",
            "CAST(NEW.id AS TEXT)",
            "'message_id=' || NEW.message_id || ';subject_id=' || NEW.subject_id || ';confidence=' || NEW.confidence",
            "'message_id=' || OLD.message_id || ';subject_id=' || OLD.subject_id || ';confidence=' || OLD.confidence",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "session_progress",
            "NEW.session_id",
            "'claimed=' || COALESCE(NEW.claimed_count, -1) || ';committed=' || COALESCE(NEW.committed_count, -1) || ';vectorized=' || COALESCE(NEW.vectorized_count, -1)",
            "'claimed=' || COALESCE(OLD.claimed_count, -1) || ';committed=' || COALESCE(OLD.committed_count, -1) || ';vectorized=' || COALESCE(OLD.vectorized_count, -1)",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "session_blacklist",
            "NEW.session_id",
            "'session_id=' || NEW.session_id",
            "'session_id=' || OLD.session_id",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "session_whitelist",
            "NEW.session_id",
            "'session_id=' || NEW.session_id",
            "'session_id=' || OLD.session_id",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "message_blacklist",
            "NEW.session_id || ':' || NEW.message_index",
            "'session_id=' || NEW.session_id || ';message_index=' || NEW.message_index",
            "'session_id=' || OLD.session_id || ';message_index=' || OLD.message_index",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "workspace_blacklist",
            "NEW.workspace_id",
            "'workspace_id=' || NEW.workspace_id",
            "'workspace_id=' || OLD.workspace_id",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "classify_failures",
            "NEW.failure_key",
            "'session_id=' || NEW.session_id || ';message_index=' || NEW.message_index || ';failure_count=' || NEW.failure_count",
            "'session_id=' || OLD.session_id || ';message_index=' || OLD.message_index || ';failure_count=' || OLD.failure_count",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
        (
            "memory_actions",
            "CAST(NEW.id AS TEXT)",
            "'action_type=' || NEW.action_type || ';subject_id=' || COALESCE(NEW.subject_id, '') || ';status=' || NEW.status || ';attempts=' || NEW.attempts",
            "'action_type=' || OLD.action_type || ';subject_id=' || COALESCE(OLD.subject_id, '') || ';status=' || OLD.status || ';attempts=' || OLD.attempts",
            "COALESCE(NEW.updated_by, NEW.created_by, 'Unknown')",
            "COALESCE(OLD.updated_by, OLD.created_by, 'Unknown')",
        ),
    ]
    # Drop legacy synthesis_queue triggers explicitly in case the table is
    # being dropped in the same migration pass.
    for op in ("insert", "update", "delete"):
        conn.execute(f"DROP TRIGGER IF EXISTS trg_synthesis_queue_audit_{op}")
    for table, record_id, new_values, old_values, new_actor, old_actor in specs:
        conn.executescript(
            f"""
            DROP TRIGGER IF EXISTS trg_{table}_audit_insert;
            DROP TRIGGER IF EXISTS trg_{table}_audit_update;
            DROP TRIGGER IF EXISTS trg_{table}_audit_delete;

            CREATE TRIGGER IF NOT EXISTS trg_{table}_audit_insert
            AFTER INSERT ON {table}
            BEGIN
                INSERT INTO audit_log(operation, table_name, record_id, old_values, new_values, changed_by)
                VALUES ('INSERT', '{table}', {record_id}, NULL, {new_values}, {new_actor});
            END;

            CREATE TRIGGER IF NOT EXISTS trg_{table}_audit_update
            AFTER UPDATE ON {table}
            BEGIN
                INSERT INTO audit_log(operation, table_name, record_id, old_values, new_values, changed_by)
                VALUES ('UPDATE', '{table}', {record_id}, {old_values}, {new_values}, {new_actor});
            END;

            CREATE TRIGGER IF NOT EXISTS trg_{table}_audit_delete
            AFTER DELETE ON {table}
            BEGIN
                INSERT INTO audit_log(operation, table_name, record_id, old_values, new_values, changed_by)
                VALUES ('DELETE', '{table}', {record_id.replace('NEW.', 'OLD.')}, {old_values}, NULL, {old_actor});
            END;
            """
        )


def _row_to_subject(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dewey_id": row["dewey_id"],
        "parent_id": row["parent_subject_id"],
        "name": row["name"],
        "description": row["description"],
        "state": row["state"],
        "message_count": row["message_count"],
        "last_activity": row["last_activity"] or "",
        "created": row["created_date_utc"],
        "created_by": row["created_by"],
        "updated_date_utc": row["updated_date_utc"],
        "updated_by": row["updated_by"],
    }


def load_subjects(root: Path | None = None) -> list[dict[str, Any]]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT * FROM semantic_subjects ORDER BY dewey_id"
        ).fetchall()
    return [_row_to_subject(row) for row in rows]


def save_subjects(subjects: list[dict[str, Any]], root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        existing = {
            row["id"]
            for row in conn.execute("SELECT id FROM semantic_subjects").fetchall()
        }
        for subject in subjects:
            conn.execute(
                """
                INSERT INTO semantic_subjects(
                    id, dewey_id, parent_subject_id, name, description, state,
                    message_count, last_activity, created_date_utc, updated_date_utc,
                    created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    dewey_id=excluded.dewey_id,
                    parent_subject_id=excluded.parent_subject_id,
                    name=excluded.name,
                    description=excluded.description,
                    state=excluded.state,
                    message_count=excluded.message_count,
                    last_activity=excluded.last_activity,
                    updated_date_utc=CURRENT_TIMESTAMP,
                    updated_by=excluded.updated_by
                """,
                (
                    subject["id"],
                    subject["dewey_id"],
                    subject.get("parent_id"),
                    subject["name"],
                    subject.get("description", ""),
                    subject["state"],
                    int(subject.get("message_count", 0)),
                    subject.get("last_activity") or None,
                    subject.get("created") or subject.get("created_date_utc") or None,
                    subject.get("created_by", ACTOR_SUBJECT_GENERATOR),
                    subject.get("updated_by", ACTOR_SUBJECT_STATE_MANAGER),
                ),
            )
        stale = existing - {subject["id"] for subject in subjects}
        if stale:
            conn.executemany(
                "DELETE FROM semantic_subjects WHERE id = ?",
                [(subject_id,) for subject_id in stale],
            )


def get_subject(subject_id: str, root: Path | None = None) -> dict[str, Any] | None:
    init_db(root)
    with connect(root) as conn:
        row = conn.execute(
            "SELECT * FROM semantic_subjects WHERE id = ? OR dewey_id = ?",
            (subject_id, subject_id),
        ).fetchone()
    return _row_to_subject(row) if row is not None else None


def create_subject_record(subject: dict[str, Any], root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            INSERT INTO semantic_subjects(
                id, dewey_id, parent_subject_id, name, description, state,
                message_count, last_activity, created_date_utc, updated_date_utc,
                created_by, updated_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
            """,
            (
                subject["id"],
                subject["dewey_id"],
                subject.get("parent_id"),
                subject["name"],
                subject.get("description", ""),
                subject["state"],
                int(subject.get("message_count", 0)),
                subject.get("last_activity"),
                subject.get("created_by", ACTOR_SUBJECT_GENERATOR),
                subject.get("updated_by", ACTOR_SUBJECT_GENERATOR),
            ),
        )


def update_subject_parent_and_dewey(
    subject_id: str,
    new_dewey_id: str,
    new_parent_subject_id: str | None,
    descendant_updates: list[tuple[str, str]],
    actor: str,
    root: Path | None = None,
) -> None:
    """Update a subject's parent and dewey_id in a single transaction.
    Also updates all descendant dewey_ids supplied in descendant_updates.
    Each entry in descendant_updates is (descendant_id, new_dewey_id).
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_subjects
            SET parent_subject_id = ?,
                dewey_id = ?,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ?
            """,
            (new_parent_subject_id, new_dewey_id, actor, subject_id),
        )
        for desc_id, desc_dewey in descendant_updates:
            conn.execute(
                """
                UPDATE semantic_subjects
                SET dewey_id = ?,
                    updated_date_utc = CURRENT_TIMESTAMP,
                    updated_by = ?
                WHERE id = ?
                """,
                (desc_dewey, actor, desc_id),
            )


def update_subject_state(subject_id: str, state: str, last_activity: str, actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_subjects
            SET state = ?,
                last_activity = ?,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ? OR dewey_id = ?
            """,
            (state, last_activity, actor, subject_id, subject_id),
        )


def increment_subject_message_count(subject_id: str, delta: int, last_activity: str, actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_subjects
            SET message_count = message_count + ?,
                last_activity = ?,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ? OR dewey_id = ?
            """,
            (delta, last_activity, actor, subject_id, subject_id),
        )


def read_subject_memory(subject_id: str, root: Path | None = None) -> str:
    init_db(root)
    with connect(root) as conn:
        row = conn.execute(
            "SELECT memory_content FROM semantic_subjects WHERE id = ? OR dewey_id = ?",
            (subject_id, subject_id),
        ).fetchone()
    return row["memory_content"] if row is not None and row["memory_content"] else ""


def write_subject_memory(subject_id: str, content: str, actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_subjects
            SET memory_content = ?,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ? OR dewey_id = ?
            """,
            (content, actor, subject_id, subject_id),
        )


def upsert_session_messages(
    session_id: str,
    messages: list[dict[str, Any]],
    workspace_id: str,
    actor: str,
    root: Path | None = None,
) -> None:
    init_db(root)
    with connect(root) as conn:
        existing_session = conn.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if existing_session is None:
            conn.execute(
                """
                INSERT INTO sessions(
                    id, workspace_id, first_message_utc, last_message_utc, message_count,
                    created_by, updated_by
                ) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?)
                """,
                (session_id, workspace_id, len(messages), actor, actor),
            )
        else:
            conn.execute(
                """
                UPDATE sessions
                SET workspace_id = ?,
                    last_message_utc = CURRENT_TIMESTAMP,
                    message_count = ?,
                    updated_date_utc = CURRENT_TIMESTAMP,
                    updated_by = ?
                WHERE id = ?
                """,
                (workspace_id, len(messages), actor, session_id),
            )

        # Pre-fetch existing IDs so re-upserts preserve the UUID already assigned.
        existing_ids: dict[int, str] = {
            int(row["message_index"]): row["id"]
            for row in conn.execute(
                "SELECT id, message_index FROM autobiographical_memory WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        }

        for idx, msg in enumerate(messages):
            role = str(msg.get("role", "user") or "user")
            if role not in {"user", "assistant", "system", "tool"}:
                role = "assistant"
            content = str(msg.get("content", ""))
            message_id = existing_ids.get(idx) or str(uuid4())
            conn.execute(
                """
                INSERT INTO autobiographical_memory(
                    id, session_id, message_index, speaker, raw_content, cleaned_content,
                    created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, message_index) DO UPDATE SET
                    speaker = excluded.speaker,
                    raw_content = excluded.raw_content,
                    cleaned_content = excluded.cleaned_content,
                    updated_date_utc = CURRENT_TIMESTAMP,
                    updated_by = excluded.updated_by
                """,
                (
                    message_id,
                    session_id,
                    idx,
                    role,
                    content,
                    content,
                    actor,
                    actor,
                ),
            )

        conn.execute(
            "DELETE FROM autobiographical_memory WHERE session_id = ? AND message_index >= ?",
            (session_id, len(messages)),
        )


def get_all_messages_for_session(
    session_id: str,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return all messages for the given session, ordered by message_index.

    Each dict has keys: id, session_id, message_index, role (from speaker),
    content (from raw_content).  This is the shape expected by classify_messages
    and the escalation helpers in server.py.
    """
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, message_index, speaker, raw_content
            FROM autobiographical_memory
            WHERE session_id = ?
            ORDER BY message_index
            """,
            (session_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "message_index": row["message_index"],
            "role": row["speaker"],
            "content": row["raw_content"],
        }
        for row in rows
    ]


def get_messages_by_ids(
    message_ids: list[str],
    root: Path | None = None,
) -> list[dict]:
    """
    Return messages for the given list of autobiographical_memory IDs, ordered
    by message_index.  IDs not found in the DB are silently omitted.

    Each dict has the same shape as get_all_messages_for_session: id,
    session_id, message_index, role, content.
    """
    if not message_ids:
        return []
    init_db(root)
    placeholders = ",".join("?" for _ in message_ids)
    with connect(root) as conn:
        rows = conn.execute(
            f"""
            SELECT id, session_id, message_index, speaker, raw_content
            FROM autobiographical_memory
            WHERE id IN ({placeholders})
            ORDER BY message_index
            """,
            tuple(message_ids),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "message_index": row["message_index"],
            "role": row["speaker"],
            "content": row["raw_content"],
        }
        for row in rows
    ]


def load_assignments(session_id: str, root: Path | None = None) -> dict[int, str | None]:
    """
    Return the primary subject (highest-confidence link) per message index for
    the given session. Messages with no link map to None.

    Backwards-compatible single-subject view of the multi-link relevance model.
    Callers that need every link should use load_all_assignments().
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT
                m.message_index,
                (
                    SELECT l.subject_id
                    FROM semantic_links AS l
                    WHERE l.message_id = m.id
                    ORDER BY l.confidence DESC, l.id ASC
                    LIMIT 1
                ) AS subject_id
            FROM autobiographical_memory AS m
            WHERE m.session_id = ?
            ORDER BY m.message_index
            """,
            (session_id,),
        ).fetchall()
    out: dict[int, str | None] = {}
    for row in rows:
        out[int(row["message_index"])] = row["subject_id"]
    return out


def load_all_assignments(
    session_id: str, root: Path | None = None
) -> dict[int, list[tuple[str, float]]]:
    """
    Return every (subject_id, confidence) link per message index for the given
    session, sorted by confidence descending. Messages with no link map to [].
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT m.message_index, l.subject_id, l.confidence
            FROM autobiographical_memory AS m
            LEFT JOIN semantic_links AS l ON l.message_id = m.id
            WHERE m.session_id = ?
            ORDER BY m.message_index, l.confidence DESC, l.id ASC
            """,
            (session_id,),
        ).fetchall()
    out: dict[int, list[tuple[str, float]]] = {}
    for row in rows:
        idx = int(row["message_index"])
        bucket = out.setdefault(idx, [])
        if row["subject_id"] is not None:
            bucket.append((str(row["subject_id"]), float(row["confidence"])))
    return out


def save_assignments(
    session_id: str,
    assignments: dict[int, Any],
    actor: str,
    root: Path | None = None,
) -> None:
    """
    Persist message-to-subject links for a session.

    Accepted value shapes per message index:
      - None          -> no links (any existing links for that message are cleared)
      - str           -> single link with confidence=1.0 (legacy single-subject form)
      - dict[str, float] -> {subject_id: confidence} for multi-subject links
      - list[tuple[str, float]] -> [(subject_id, confidence), ...]

    Confidences are clamped to [0.0, 1.0]. Links to subject ids not present in
    semantic_subjects are skipped with a warning.
    """
    init_db(root)
    with connect(root) as conn:
        message_rows = conn.execute(
            "SELECT id, message_index FROM autobiographical_memory WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        id_by_index = {int(row["message_index"]): row["id"] for row in message_rows}
        if not id_by_index:
            return

        # Scope the DELETE to only messages actually present in the incoming
        # `assignments` dict. A sparse call (e.g. a single batch) must not
        # wipe links for messages outside that batch.
        in_scope_message_ids = tuple(
            id_by_index[idx] for idx in assignments.keys() if idx in id_by_index
        )
        if in_scope_message_ids:
            placeholders = ",".join("?" for _ in in_scope_message_ids)
            conn.execute(
                f"DELETE FROM semantic_links WHERE message_id IN ({placeholders})",
                in_scope_message_ids,
            )

        valid_subject_ids: set[str] = {
            row["id"]
            for row in conn.execute("SELECT id FROM semantic_subjects").fetchall()
        }

        for idx, message_id in id_by_index.items():
            raw = assignments.get(idx)
            pairs: list[tuple[str, float]] = []
            if raw is None:
                continue
            if isinstance(raw, str):
                pairs.append((raw, 1.0))
            elif isinstance(raw, dict):
                for sid, score in raw.items():
                    try:
                        pairs.append((str(sid), float(score)))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(raw, (list, tuple)):
                for entry in raw:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        try:
                            pairs.append((str(entry[0]), float(entry[1])))
                        except (TypeError, ValueError):
                            continue
            else:
                continue

            for sid, score in pairs:
                if sid not in valid_subject_ids:
                    _log.warning(
                        "save_assignments: subject_id %r not in semantic_subjects - skipping link for message %s",
                        sid, message_id,
                    )
                    continue
                conf = max(0.0, min(1.0, score))
                conn.execute(
                    """
                    INSERT INTO semantic_links(
                        message_id, subject_id, confidence,
                        created_by, updated_by
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(message_id, subject_id) DO UPDATE SET
                        confidence = excluded.confidence,
                        updated_date_utc = CURRENT_TIMESTAMP,
                        updated_by = excluded.updated_by
                    """,
                    (message_id, sid, conf, actor, actor),
                )


def list_sessions_for_subject(subject_id: str, root: Path | None = None) -> list[str]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT m.session_id
            FROM semantic_links AS l
            INNER JOIN autobiographical_memory AS m
                ON m.id = l.message_id
            WHERE l.subject_id = ?
            ORDER BY m.session_id
            """,
            (subject_id,),
        ).fetchall()
    return [str(row["session_id"]) for row in rows]


# ---------------------------------------------------------------------------
# Confidence attenuation
# ---------------------------------------------------------------------------

def get_links_eligible_for_attenuation(
    cutoff_days: int = 7,
    root: Path | None = None,
) -> list[dict]:
    """
    Return semantic_links rows that have not been updated in the past
    `cutoff_days` days and are therefore eligible for confidence attenuation.
    Ordered oldest-first so the daemon drains the most stale links first.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, message_id, subject_id, confidence
            FROM semantic_links
            WHERE updated_date_utc < datetime('now', ? || ' days')
            ORDER BY updated_date_utc ASC
            """,
            (f"-{cutoff_days}",),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "message_id": str(r["message_id"]),
            "subject_id": str(r["subject_id"]),
            "confidence": float(r["confidence"]),
        }
        for r in rows
    ]


def attenuate_link_confidence(
    link_id: int,
    new_confidence: float,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Write the new confidence score for a semantic_links row and reset its
    updated_date_utc.  Resetting the timestamp restarts the attenuation
    eligibility timer - the link will not be eligible again until another
    `cutoff_days` have elapsed without a touch.
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_links
            SET confidence = ?,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ?
            """,
            (new_confidence, actor, link_id),
        )


def get_links_pending_reconciliation(
    root: Path | None = None,
) -> list[dict]:
    """
    Return semantic_links rows where reconciled_at IS NULL, ordered by
    created_date_utc ascending (oldest unreconciled first).

    Each returned dict has: id, message_id, subject_id, confidence,
    created_date_utc.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, message_id, subject_id, confidence, created_date_utc
            FROM semantic_links
            WHERE reconciled_at IS NULL
            ORDER BY created_date_utc ASC
            """
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "message_id": str(r["message_id"]),
            "subject_id": str(r["subject_id"]),
            "confidence": float(r["confidence"]),
            "created_date_utc": str(r["created_date_utc"]),
        }
        for r in rows
    ]


def get_subjects_needing_split(
    threshold: int,
    root: Path | None = None,
) -> list[dict]:
    """
    Return open or dormant subjects whose message_count is at or above
    `threshold`.  Used by daemon_split._WATCH_SPEC scanner for recovery
    detection.  The watcher infrastructure deduplicates via
    has_pending_memory_action before enqueuing, so this query does not
    need to exclude subjects that already have a pending action.
    Each returned dict has: id, name, message_count.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, name, message_count
            FROM semantic_subjects
            WHERE state NOT IN ('archived', 'split')
              AND message_count >= ?
            ORDER BY message_count DESC
            """,
            (threshold,),
        ).fetchall()
    return [
        {
            "id": str(r["id"]),
            "name": str(r["name"]),
            "message_count": int(r["message_count"]),
        }
        for r in rows
    ]


def get_archived_subjects_needing_synthesis(
    root: Path | None = None,
) -> list[dict]:
    """
    Return archived subjects that have not yet been synthesized
    (synthesized_at IS NULL).  Used by daemon_synthesize._WATCH_SPEC scanner
    for recovery detection.
    Each returned dict has: id, name.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, name
            FROM semantic_subjects
            WHERE state = 'archived'
              AND synthesized_at IS NULL
            ORDER BY updated_date_utc ASC
            """,
        ).fetchall()
    return [
        {
            "id": str(r["id"]),
            "name": str(r["name"]),
        }
        for r in rows
    ]


def mark_subject_synthesized(
    subject_id: str,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Stamp synthesized_at on a semantic_subjects row after a successful
    finalize_subject run.  Prevents the synthesize scanner from re-queueing
    the same subject on subsequent passes.
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_subjects
            SET synthesized_at = CURRENT_TIMESTAMP,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ?
            """,
            (actor, subject_id),
        )


def mark_link_reconciled(
    link_id: int,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Stamp reconciled_at on a semantic_links row to indicate it has been
    processed by daemon_reconcile. Idempotent: re-stamping an already-
    reconciled row just updates the timestamp.
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            UPDATE semantic_links
            SET reconciled_at = CURRENT_TIMESTAMP,
                updated_date_utc = CURRENT_TIMESTAMP,
                updated_by = ?
            WHERE id = ?
            """,
            (actor, link_id),
        )


def insert_subject_relation(
    from_link_id: int,
    to_link_id: int,
    subject_id: str,
    relation: str,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Insert a relation triple into subject_relations.

    from_link_id - the older link being evaluated
    to_link_id   - the newer link (the one just reconciled)
    relation     - one of 'supports', 'contradicts', 'updates'
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            """
            INSERT INTO subject_relations (subject_id, from_link_id, to_link_id, relation, created_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subject_id, from_link_id, to_link_id, relation, actor),
        )


def get_subject_relations_for_subject(
    subject_id: str,
    root: Path | None = None,
) -> list[dict]:
    """
    Return all subject_relations rows for a subject, joined with semantic_links
    to include each endpoint's current confidence value.

    Because subject_relations has ON DELETE CASCADE FKs on both from_link_id
    and to_link_id, every returned row is guaranteed to have live endpoints.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT
                sr.id,
                sr.subject_id,
                sr.from_link_id,
                sr.to_link_id,
                sr.relation,
                sr.created_date_utc,
                fl.confidence AS from_confidence,
                fl.message_id AS from_message_id,
                tl.confidence AS to_confidence,
                tl.message_id AS to_message_id
            FROM subject_relations sr
            JOIN semantic_links fl ON fl.id = sr.from_link_id
            JOIN semantic_links tl ON tl.id = sr.to_link_id
            WHERE sr.subject_id = ?
            ORDER BY sr.created_date_utc ASC
            """,
            (subject_id,),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "subject_id": str(r["subject_id"]),
            "from_link_id": int(r["from_link_id"]),
            "to_link_id": int(r["to_link_id"]),
            "relation": str(r["relation"]),
            "created_date_utc": str(r["created_date_utc"]),
            "from_confidence": float(r["from_confidence"]),
            "from_message_id": str(r["from_message_id"]),
            "to_confidence": float(r["to_confidence"]),
            "to_message_id": str(r["to_message_id"]),
        }
        for r in rows
    ]


def delete_semantic_link(
    link_id: int,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Delete a semantic_links row by its integer primary key id.
    The audit trigger captures the DELETE for the record.
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            "UPDATE semantic_links SET updated_by = ?, updated_date_utc = CURRENT_TIMESTAMP WHERE id = ?",
            (actor, link_id),
        )
        conn.execute("DELETE FROM semantic_links WHERE id = ?", (link_id,))


# All progress columns tracked in session_progress.  Order is significant:
# the first two are the classify pipeline; vectorized_count is optional.
_ALL_PROGRESS_COLUMNS: tuple[str, ...] = ("claimed_count", "committed_count", "vectorized_count")


def load_claimed_counts(root: Path | None = None) -> dict[str, int]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT session_id, claimed_count FROM session_progress WHERE claimed_count IS NOT NULL"
        ).fetchall()
    return {str(row["session_id"]): int(row["claimed_count"]) for row in rows}


def save_claimed_counts(counts: dict[str, int], actor: str, root: Path | None = None) -> None:
    _sync_session_progress(counts, "claimed_count", actor, root)


def load_committed_counts(root: Path | None = None) -> dict[str, int]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT session_id, committed_count FROM session_progress WHERE committed_count IS NOT NULL"
        ).fetchall()
    return {str(row["session_id"]): int(row["committed_count"]) for row in rows}


def save_committed_counts(counts: dict[str, int], actor: str, root: Path | None = None) -> None:
    _sync_session_progress(counts, "committed_count", actor, root)


def get_session_message_counts(root: Path | None = None) -> dict[str, int]:
    """Return {session_id: total_message_count} for all sessions with AB records."""
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT session_id, COUNT(*) AS cnt FROM autobiographical_memory GROUP BY session_id"
        ).fetchall()
    return {str(row["session_id"]): int(row["cnt"]) for row in rows}


def load_vectorized_counts(root: Path | None = None) -> dict[str, int]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT session_id, vectorized_count FROM session_progress WHERE vectorized_count IS NOT NULL"
        ).fetchall()
    return {str(row["session_id"]): int(row["vectorized_count"]) for row in rows}


def save_vectorized_counts(counts: dict[str, int], actor: str, root: Path | None = None) -> None:
    _sync_session_progress(counts, "vectorized_count", actor, root)


def _sync_session_progress(counts: dict[str, int], column: str, actor: str, root: Path | None = None) -> None:
    init_db(root)
    # Build the list of sibling columns so we can decide whether a row can be
    # fully deleted (all siblings are NULL) or must only be NULLed out.
    others = [c for c in _ALL_PROGRESS_COLUMNS if c != column]
    select_cols = ", ".join(["session_id"] + others)
    with connect(root) as conn:
        rows = conn.execute(f"SELECT {select_cols} FROM session_progress").fetchall()
        existing = {str(row["session_id"]) for row in rows}
        sibling_values: dict[str, dict[str, object]] = {
            str(row["session_id"]): {c: row[c] for c in others} for row in rows
        }
        for session_id, count in counts.items():
            conn.execute(
                f"""
                INSERT INTO session_progress(session_id, {column}, created_by, updated_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    {column} = excluded.{column},
                    updated_date_utc = CURRENT_TIMESTAMP,
                    updated_by = excluded.updated_by
                """,
                (session_id, int(count), actor, actor),
            )
        for session_id in existing - set(counts.keys()):
            all_siblings_null = all(
                sibling_values.get(session_id, {}).get(c) is None for c in others
            )
            if all_siblings_null:
                conn.execute("DELETE FROM session_progress WHERE session_id = ?", (session_id,))
            else:
                conn.execute(
                    f"""
                    UPDATE session_progress
                    SET {column} = NULL,
                        updated_date_utc = CURRENT_TIMESTAMP,
                        updated_by = ?
                    WHERE session_id = ?
                    """,
                    (actor, session_id),
                )


def load_workspace_blacklist(root: Path | None = None) -> set[str]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute("SELECT workspace_id FROM workspace_blacklist").fetchall()
    return {str(row["workspace_id"]) for row in rows}


def save_workspace_blacklist(blacklist: set[str], actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute("DELETE FROM workspace_blacklist")
        conn.executemany(
            """
            INSERT INTO workspace_blacklist(workspace_id, created_by, updated_by)
            VALUES (?, ?, ?)
            """,
            [(workspace_id, actor, actor) for workspace_id in sorted(blacklist)],
        )


def load_session_blacklist(root: Path | None = None) -> set[str]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute("SELECT session_id FROM session_blacklist").fetchall()
    return {str(row["session_id"]) for row in rows}


def save_session_blacklist(blacklist: set[str], actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute("DELETE FROM session_blacklist")
        conn.executemany(
            """
            INSERT INTO session_blacklist(session_id, created_by, updated_by)
            VALUES (?, ?, ?)
            """,
            [(session_id, actor, actor) for session_id in sorted(blacklist)],
        )


def load_session_whitelist(root: Path | None = None) -> set[str]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute("SELECT session_id FROM session_whitelist").fetchall()
    return {str(row["session_id"]) for row in rows}


def save_session_whitelist(whitelist: set[str], actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute("DELETE FROM session_whitelist")
        conn.executemany(
            """
            INSERT INTO session_whitelist(session_id, created_by, updated_by)
            VALUES (?, ?, ?)
            """,
            [(session_id, actor, actor) for session_id in sorted(whitelist)],
        )


def load_message_blacklist(root: Path | None = None) -> dict[str, dict[int, str]]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT session_id, message_index, reason FROM message_blacklist"
            " WHERE status = 'blacklisted'"
            " ORDER BY session_id, message_index"
        ).fetchall()
    out: dict[str, dict[int, str]] = {}
    for row in rows:
        out.setdefault(str(row["session_id"]), {})[int(row["message_index"])] = str(row["reason"])
    return out


def save_message_blacklist(blacklist: dict[str, dict[int, str]], actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute("DELETE FROM message_blacklist")
        rows = [
            (session_id, int(idx), "blacklisted", reason, actor, actor)
            for session_id, index_map in blacklist.items()
            for idx, reason in sorted(index_map.items())
        ]
        if rows:
            conn.executemany(
                """
                INSERT INTO message_blacklist(
                    session_id, message_index, status, reason, created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def set_blacklist_status(
    status: str,
    reason: str | None = None,
    session_id: str | None = None,
    root: Path | None = None,
) -> int:
    """
    Update the status (and optionally reason) of message_blacklist rows.
    When session_id is provided, only that session's rows are updated.
    Returns the number of rows changed.
    """
    init_db(root)
    rowcount = 0
    with connect(root) as conn:
        if reason is not None and session_id is not None:
            rowcount = conn.execute(
                "UPDATE message_blacklist SET status = ?, reason = ? WHERE session_id = ?",
                (status, reason, session_id),
            ).rowcount
        elif reason is not None:
            rowcount = conn.execute(
                "UPDATE message_blacklist SET status = ?, reason = ?",
                (status, reason),
            ).rowcount
        elif session_id is not None:
            rowcount = conn.execute(
                "UPDATE message_blacklist SET status = ? WHERE session_id = ?",
                (status, session_id),
            ).rowcount
        else:
            rowcount = conn.execute(
                "UPDATE message_blacklist SET status = ?",
                (status,),
            ).rowcount
    return rowcount


def load_classify_failures(root: Path | None = None) -> dict[str, int]:
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            "SELECT failure_key, failure_count FROM classify_failures"
        ).fetchall()
    return {str(row["failure_key"]): int(row["failure_count"]) for row in rows}


def save_classify_failures(failures: dict[str, int], actor: str, root: Path | None = None) -> None:
    init_db(root)
    with connect(root) as conn:
        conn.execute("DELETE FROM classify_failures")
        rows = []
        for failure_key, count in failures.items():
            session_id, _, suffix = failure_key.partition(":")
            rows.append((failure_key, session_id, int(suffix or 0), int(count), actor, actor))
        if rows:
            conn.executemany(
                """
                INSERT INTO classify_failures(
                    failure_key, session_id, message_index, failure_count,
                    created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def enqueue_memory_action(
    action_type: str,
    subject_id: str | None,
    payload: dict[str, Any] | None = None,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> int:
    """
    Insert a new pending action and return its row id.
    """
    import json as _json
    init_db(root)
    payload_text = _json.dumps(payload or {}, ensure_ascii=False)
    with connect(root) as conn:
        cur = conn.execute(
            """
            INSERT INTO memory_actions(
                action_type, subject_id, payload_json, status,
                attempts, created_by, updated_by
            ) VALUES (?, ?, ?, 'pending', 0, ?, ?)
            """,
            (action_type, subject_id, payload_text, actor, actor),
        )
        return int(cur.lastrowid or 0)


def claim_next_memory_action(
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
    action_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Atomically claim the oldest pending action by flipping its status to
    'running' and incrementing attempts. Returns the claimed row as a dict
    (with payload parsed), or None if the queue is empty.

    Pass action_types to restrict which action types this daemon will claim.
    Any pending action whose type is not in the list is left untouched.
    """
    import json as _json
    init_db(root)
    with connect(root) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if action_types:
            placeholders = ",".join("?" * len(action_types))
            row = conn.execute(
                f"""
                SELECT id, action_type, subject_id, payload_json, attempts
                FROM memory_actions
                WHERE status = 'pending'
                  AND action_type IN ({placeholders})
                ORDER BY id ASC
                LIMIT 1
                """,
                action_types,
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id, action_type, subject_id, payload_json, attempts
                FROM memory_actions
                WHERE status = 'pending'
                ORDER BY id ASC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        action_id = int(row["id"])
        conn.execute(
            """
            UPDATE memory_actions
            SET status = 'running',
                attempts = attempts + 1,
                updated_by = ?,
                updated_date_utc = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (actor, action_id),
        )
        conn.execute("COMMIT")
        try:
            payload = _json.loads(row["payload_json"] or "{}")
        except (ValueError, TypeError):
            payload = {}
        return {
            "id": action_id,
            "action_type": str(row["action_type"]),
            "subject_id": (str(row["subject_id"]) if row["subject_id"] is not None else None),
            "payload": payload,
            "attempts": int(row["attempts"]) + 1,
        }


def complete_memory_action(
    action_id: int,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Delete the action row on successful completion. Audit log retains the
    full INSERT/UPDATE/DELETE history.
    """
    init_db(root)
    with connect(root) as conn:
        # Touch updated_by first so the DELETE audit row attributes the change
        # to the actor that completed the work.
        conn.execute(
            "UPDATE memory_actions SET updated_by = ?, updated_date_utc = CURRENT_TIMESTAMP WHERE id = ?",
            (actor, action_id),
        )
        conn.execute("DELETE FROM memory_actions WHERE id = ?", (action_id,))


def fail_memory_action(
    action_id: int,
    error: str,
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> str:
    """
    Record a failure for the running action. If attempts is below
    MEMORY_ACTION_MAX_ATTEMPTS the row goes back to 'pending' for retry;
    otherwise it transitions to 'failed' and is retained for later
    re-processing. Returns the resulting status.
    """
    init_db(root)
    with connect(root) as conn:
        row = conn.execute(
            "SELECT attempts FROM memory_actions WHERE id = ?",
            (action_id,),
        ).fetchone()
        if row is None:
            return "missing"
        new_status = (
            MEMORY_ACTION_STATUS_FAILED
            if int(row["attempts"]) >= MEMORY_ACTION_MAX_ATTEMPTS
            else MEMORY_ACTION_STATUS_PENDING
        )
        conn.execute(
            """
            UPDATE memory_actions
            SET status = ?,
                last_error = ?,
                updated_by = ?,
                updated_date_utc = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (new_status, error, actor, action_id),
        )
        return new_status


def peek_pending_memory_actions(root: Path | None = None) -> list[dict[str, Any]]:
    """
    Introspection helper used by tests and diagnostics. Returns all rows
    (any status) ordered by id, with payload still as raw text.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, action_type, subject_id, payload_json, status,
                   attempts, last_error
            FROM memory_actions
            ORDER BY id ASC
            """
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "action_type": str(r["action_type"]),
            "subject_id": (str(r["subject_id"]) if r["subject_id"] is not None else None),
            "payload_json": str(r["payload_json"]),
            "status": str(r["status"]),
            "attempts": int(r["attempts"]),
            "last_error": (str(r["last_error"]) if r["last_error"] is not None else None),
        }
        for r in rows
    ]


def has_pending_memory_action(
    action_type: str,
    subject_id: str,
    root: Path | None = None,
) -> bool:
    """
    Return True if a memory_actions row with the given action_type and
    subject_id exists in status 'pending' or 'running'. Used as the dedupe
    guard before enqueuing a new action to avoid double-queueing the same
    work unit.
    """
    init_db(root)
    with connect(root) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM memory_actions
            WHERE action_type = ? AND subject_id IS ? AND status IN ('pending', 'running')
            LIMIT 1
            """,
            (action_type, subject_id),
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Context compaction queries
# ---------------------------------------------------------------------------

def get_subjects_by_lru(
    session_id: str,
    root: Path | None = None,
) -> list[dict]:
    """
    Return subjects that have semantic links in the given session, ordered by
    MAX(message_index) ASC (coldest-last-touched first).

    Each row is a dict with:
      subject_id   - the subject UUID
      last_msg_idx - the highest message_index among all links for that subject
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT sl.subject_id, MAX(am.message_index) AS last_msg_idx
            FROM semantic_links sl
            JOIN autobiographical_memory am ON am.id = sl.message_id
            WHERE am.session_id = ?
            GROUP BY sl.subject_id
            ORDER BY last_msg_idx ASC
            """,
            (session_id,),
        ).fetchall()
    return [
        {"subject_id": str(r["subject_id"]), "last_msg_idx": int(r["last_msg_idx"])}
        for r in rows
    ]


def get_messages_fully_covered(
    session_id: str,
    closing_ids: set[str],
    root: Path | None = None,
) -> set[int]:
    """
    Return the set of message_index values (for the given session) where every
    subject linked to that message is inside closing_ids.

    Unlinked messages (no semantic_links rows at all) are never returned.
    Messages that have at least one link to a subject outside closing_ids
    are also excluded.
    """
    if not closing_ids:
        return set()
    init_db(root)
    placeholders = ",".join("?" * len(closing_ids))
    params: list = [session_id, session_id] + list(closing_ids)
    with connect(root) as conn:
        rows = conn.execute(
            f"""
            SELECT am.message_index
            FROM autobiographical_memory am
            WHERE am.session_id = ?
              AND EXISTS (
                  SELECT 1 FROM semantic_links sl WHERE sl.message_id = am.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM semantic_links sl
                  JOIN autobiographical_memory am2 ON am2.id = sl.message_id
                  WHERE am2.session_id = ?
                    AND am2.id = am.id
                    AND sl.subject_id NOT IN ({placeholders})
              )
            """,
            params,
        ).fetchall()
    return {int(r["message_index"]) for r in rows}


# ---------------------------------------------------------------------------
# Optional vector support (sqlite-vec)
# ---------------------------------------------------------------------------

def _maybe_create_vec_table(conn: sqlite3.Connection) -> None:
    """
    Create the vec_messages virtual table if sqlite-vec is installed.
    Called from init_db(); silently skips if the extension is unavailable.
    """
    import embeddings as _emb  # lazy import - embeddings is optional
    if not _emb.is_available():
        return
    if not _emb.load_extension(conn):
        return
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(embedding float[768])"
        )
    except sqlite3.OperationalError as exc:
        _log.debug("Could not create vec_messages table: %s", exc)
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_subjects USING vec0(embedding float[768])"
        )
    except sqlite3.OperationalError as exc:
        _log.debug("Could not create vec_subjects table: %s", exc)


def connect_vec(root: Path | None = None) -> tuple[sqlite3.Connection, bool]:
    """
    Open a connection with the sqlite-vec extension loaded.
    Returns (conn, True) on success, (conn, False) if extension unavailable.
    Caller is responsible for closing the connection.
    """
    import embeddings as _emb
    conn = connect(root)
    loaded = _emb.load_extension(conn)
    return conn, loaded


def get_messages_needing_embedding(
    session_id: str,
    from_index: int,
    limit: int,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return up to `limit` messages from `session_id` whose message_index is
    >= `from_index`, ordered by message_index ascending.
    """
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT rowid, id, message_index, speaker, raw_content
            FROM autobiographical_memory
            WHERE session_id = ?
              AND message_index >= ?
            ORDER BY message_index
            LIMIT ?
            """,
            (session_id, from_index, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_message_embeddings_batch(
    embeddings_by_rowid: dict[int, bytes],
    root: Path | None = None,
) -> int:
    """
    Insert or replace embeddings in vec_messages for multiple rows at once.
    Returns the count of successfully upserted embeddings.
    """
    if not embeddings_by_rowid:
        return 0
    conn, loaded = connect_vec(root)
    if not loaded:
        return 0
    count = 0
    try:
        for rowid, embedding_bytes in embeddings_by_rowid.items():
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_messages(rowid, embedding) VALUES (?, ?)",
                    (rowid, embedding_bytes),
                )
                count += 1
            except Exception as exc:
                _log.debug("vec upsert failed for rowid=%d: %s", rowid, exc)
        conn.commit()
    except Exception as exc:
        _log.debug("vec batch upsert failed: %s", exc)
    return count


def semantic_search_messages(
    embedding_bytes: bytes,
    k: int = 10,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    KNN search over vec_messages, joined back to autobiographical_memory
    and semantic_subjects (via the highest-confidence semantic_link).

    Returns up to k results sorted by cosine distance (nearest first), each as
    a dict with keys: id, session_id, message_index, speaker, raw_content,
    timestamp_utc, distance, subject_name.
    """
    conn, loaded = connect_vec(root)
    if not loaded:
        return []
    try:
        rows = conn.execute(
            """
            WITH knn AS (
                SELECT rowid, distance
                FROM vec_messages
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            )
            SELECT
                am.id,
                am.session_id,
                am.message_index,
                am.speaker,
                am.raw_content,
                am.timestamp_utc,
                knn.distance,
                ss.name AS subject_name,
                sl.created_date_utc AS link_created_date
            FROM knn
            JOIN autobiographical_memory am ON am.rowid = knn.rowid
            LEFT JOIN semantic_links sl ON sl.id = (
                SELECT l2.id FROM semantic_links l2
                WHERE l2.message_id = am.id
                ORDER BY l2.confidence DESC, l2.id ASC
                LIMIT 1
            )
            LEFT JOIN semantic_subjects ss ON ss.id = sl.subject_id
            ORDER BY knn.distance
            """,
            (embedding_bytes, k),
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        _log.debug("semantic_search_messages failed: %s", exc)
        return []


def semantic_search_subject_messages(
    embedding_bytes: bytes,
    subject_id: str,
    k: int = 10,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    KNN search over vec_messages scoped to messages linked to a single subject.

    Used by daemon_reconcile to find messages within a subject that are
    semantically close to a new arrival. Returns up to k results sorted by
    cosine distance (nearest first), each as a dict with keys:
    id, session_id, message_index, speaker, raw_content, timestamp_utc,
    distance, link_id, link_created_date, link_confidence.
    """
    conn, loaded = connect_vec(root)
    if not loaded:
        return []
    try:
        rows = conn.execute(
            """
            WITH subject_rowids AS (
                SELECT am.rowid AS msg_rowid
                FROM semantic_links sl
                JOIN autobiographical_memory am ON am.id = sl.message_id
                WHERE sl.subject_id = ?
            ),
            knn AS (
                SELECT rowid, distance
                FROM vec_messages
                WHERE embedding MATCH ?
                AND rowid IN (SELECT msg_rowid FROM subject_rowids)
                ORDER BY distance
                LIMIT ?
            )
            SELECT
                am.id,
                am.session_id,
                am.message_index,
                am.speaker,
                am.raw_content,
                am.timestamp_utc,
                knn.distance,
                sl.id AS link_id,
                sl.created_date_utc AS link_created_date,
                sl.confidence AS link_confidence
            FROM knn
            JOIN autobiographical_memory am ON am.rowid = knn.rowid
            JOIN semantic_links sl ON sl.message_id = am.id AND sl.subject_id = ?
            ORDER BY knn.distance
            """,
            (subject_id, embedding_bytes, k, subject_id),
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        _log.debug("semantic_search_subject_messages failed: %s", exc)
        return []
    finally:
        conn.close()


def get_subject_messages_for_finalize(
    subject_id: str,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return all autobiographical_memory messages linked to a subject, ordered by
    semantic_links.confidence DESC then timestamp_utc DESC (most relevant and
    most recent first).

    Used by daemon_synthesize to build a bounded, ranked input payload for the
    finalize operation. Each returned dict has keys: message_id, speaker,
    raw_content, timestamp_utc, confidence.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT
                am.id AS message_id,
                am.speaker,
                am.raw_content,
                am.timestamp_utc,
                sl.confidence
            FROM semantic_links sl
            JOIN autobiographical_memory am ON am.id = sl.message_id
            WHERE sl.subject_id = ?
            ORDER BY sl.confidence DESC, am.timestamp_utc DESC
            """,
            (subject_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_subjects_needing_embedding(
    limit: int,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return subjects whose memory_content is non-empty and either has never been
    embedded (no row in subject_embeddings) or whose updated_date_utc is newer
    than the embedded_date_utc of the existing row.

    Returns up to `limit` rows, each with keys: id, memory_content.
    """
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT ss.id, ss.memory_content
            FROM semantic_subjects ss
            WHERE ss.memory_content != ''
              AND (
                  NOT EXISTS (
                      SELECT 1 FROM subject_embeddings se
                      WHERE se.subject_id = ss.id
                  )
                  OR ss.updated_date_utc > (
                      SELECT se.embedded_date_utc FROM subject_embeddings se
                      WHERE se.subject_id = ss.id
                  )
              )
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_subject_embedding(
    subject_id: str,
    embedding_bytes: bytes,
    root: Path | None = None,
) -> bool:
    """
    Re-embed a subject's synthesized memory.

    Deletes any existing vec_subjects row and subject_embeddings row for
    subject_id, inserts a fresh subject_embeddings row, and inserts the new
    embedding into vec_subjects keyed by the new rowid.

    Returns True on success, False if sqlite-vec is unavailable.
    """
    conn, loaded = connect_vec(root)
    if not loaded:
        return False
    try:
        existing = conn.execute(
            "SELECT rowid FROM subject_embeddings WHERE subject_id = ?",
            (subject_id,),
        ).fetchall()
        for row in existing:
            try:
                conn.execute(
                    "DELETE FROM vec_subjects WHERE rowid = ?",
                    (row["rowid"],),
                )
            except Exception as exc:
                _log.debug("vec_subjects delete failed for rowid=%s: %s", row["rowid"], exc)
        conn.execute(
            "DELETE FROM subject_embeddings WHERE subject_id = ?",
            (subject_id,),
        )
        cursor = conn.execute(
            "INSERT INTO subject_embeddings(subject_id) VALUES (?)",
            (subject_id,),
        )
        new_rowid = cursor.lastrowid
        conn.execute(
            "INSERT INTO vec_subjects(rowid, embedding) VALUES (?, ?)",
            (new_rowid, embedding_bytes),
        )
        conn.commit()
        return True
    except Exception as exc:
        _log.debug("upsert_subject_embedding failed for subject %s: %s", subject_id, exc)
        return False
    finally:
        conn.close()


def semantic_search_subjects(
    embedding_bytes: bytes,
    k: int = 10,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    KNN search over vec_subjects, joined back to semantic_subjects via
    subject_embeddings. Returns up to k results sorted by cosine distance
    (nearest first), each as a dict with keys: id, name, dewey_id,
    memory_content, distance.
    """
    conn, loaded = connect_vec(root)
    if not loaded:
        return []
    try:
        rows = conn.execute(
            """
            WITH knn AS (
                SELECT rowid, distance
                FROM vec_subjects
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            )
            SELECT
                ss.id,
                ss.name,
                ss.dewey_id,
                ss.memory_content,
                knn.distance
            FROM knn
            JOIN subject_embeddings se ON se.rowid = knn.rowid
            JOIN semantic_subjects ss ON ss.id = se.subject_id
            ORDER BY knn.distance
            """,
            (embedding_bytes, k),
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        _log.debug("semantic_search_subjects failed: %s", exc)
        return []
    finally:
        conn.close()


def get_messages_in_range(
    start_utc: str,
    end_utc: str,
    session_id: str | None = None,
    limit: int = 100,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return messages whose timestamp_utc falls in [start_utc, end_utc],
    ordered chronologically. Optional session_id narrows to a single session.

    Timestamps are compared as ISO 8601 strings (SQLite lexicographic sort works
    correctly for UTC ISO format).
    """
    with connect(root) as conn:
        if session_id is not None:
            rows = conn.execute(
                """
                SELECT am.id, am.session_id, am.message_index,
                       am.speaker, am.raw_content, am.timestamp_utc,
                       ss.name AS subject_name
                FROM autobiographical_memory am
                LEFT JOIN semantic_links sl ON sl.id = (
                    SELECT l2.id FROM semantic_links l2
                    WHERE l2.message_id = am.id
                    ORDER BY l2.confidence DESC, l2.id ASC
                    LIMIT 1
                )
                LEFT JOIN semantic_subjects ss ON ss.id = sl.subject_id
                WHERE am.timestamp_utc >= ?
                  AND am.timestamp_utc <= ?
                  AND am.session_id = ?
                ORDER BY am.timestamp_utc, am.message_index
                LIMIT ?
                """,
                (start_utc, end_utc, session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT am.id, am.session_id, am.message_index,
                       am.speaker, am.raw_content, am.timestamp_utc,
                       ss.name AS subject_name
                FROM autobiographical_memory am
                LEFT JOIN semantic_links sl ON sl.id = (
                    SELECT l2.id FROM semantic_links l2
                    WHERE l2.message_id = am.id
                    ORDER BY l2.confidence DESC, l2.id ASC
                    LIMIT 1
                )
                LEFT JOIN semantic_subjects ss ON ss.id = sl.subject_id
                WHERE am.timestamp_utc >= ?
                  AND am.timestamp_utc <= ?
                ORDER BY am.timestamp_utc, am.session_id, am.message_index
                LIMIT ?
                """,
                (start_utc, end_utc, limit),
            ).fetchall()
    return [dict(row) for row in rows]


def get_message_by_id(
    message_id: str,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """
    Return a single message by its TEXT id column, with all subject associations
    concatenated. Returns None if the id does not exist.
    """
    with connect(root) as conn:
        row = conn.execute(
            """
            SELECT am.id, am.session_id, am.message_index,
                   am.speaker, am.raw_content, am.timestamp_utc,
                   GROUP_CONCAT(ss.name, ' | ') AS subject_names
            FROM autobiographical_memory am
            LEFT JOIN semantic_links sl ON sl.message_id = am.id
            LEFT JOIN semantic_subjects ss ON ss.id = sl.subject_id
            WHERE am.id = ?
            GROUP BY am.id
            """,
            (message_id,),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# daemon_reclass helpers
# ---------------------------------------------------------------------------

def get_split_subjects(root: Path | None = None) -> list[dict]:
    """
    Return subjects with state='split', ordered by updated_date_utc ascending
    (oldest first so the daemon drains the most-stale splits first).
    Each dict has: id, name.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT id, name
            FROM semantic_subjects
            WHERE state = 'split'
            ORDER BY updated_date_utc ASC
            """
        ).fetchall()
    return [{"id": str(r["id"]), "name": str(r["name"])} for r in rows]


def get_children_of_subject(parent_id: str, root: Path | None = None) -> list[dict]:
    """
    Return direct children of the given parent subject, ordered by dewey_id.
    Each dict is the full subject shape produced by _row_to_subject.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM semantic_subjects
            WHERE parent_subject_id = ?
            ORDER BY dewey_id
            """,
            (parent_id,),
        ).fetchall()
    return [_row_to_subject(r) for r in rows]


def get_messages_linked_to_subject(subject_id: str, root: Path | None = None) -> list[dict]:
    """
    Return all autobiographical_memory rows that are linked to subject_id via
    semantic_links, ordered by message_index.

    Each dict has: id, session_id, message_index, role, content.
    This is the same shape as get_all_messages_for_session and is the format
    expected by meta_agent.classify_messages.
    """
    init_db(root)
    with connect(root) as conn:
        rows = conn.execute(
            """
            SELECT am.id, am.session_id, am.message_index,
                   am.speaker, am.raw_content
            FROM autobiographical_memory am
            INNER JOIN semantic_links sl ON sl.message_id = am.id
            WHERE sl.subject_id = ?
            ORDER BY am.message_index
            """,
            (subject_id,),
        ).fetchall()
    return [
        {
            "id": str(r["id"]),
            "session_id": str(r["session_id"]),
            "message_index": int(r["message_index"]),
            "role": str(r["speaker"]),
            "content": str(r["raw_content"]),
        }
        for r in rows
    ]


def reattribute_links(
    old_subject_id: str,
    assignments: dict[str, str],
    actor: str = ACTOR_MEMORY_ACTIONS,
    root: Path | None = None,
) -> None:
    """
    Delete all semantic_links pointing to old_subject_id and insert new links
    based on assignments.

    assignments: {message_id -> new_subject_id}

    All messages previously linked to old_subject_id have their links deleted,
    then new links to the assigned child subjects are created at confidence 1.0.
    Messages not present in assignments lose their links and are not re-linked.
    """
    init_db(root)
    with connect(root) as conn:
        conn.execute(
            "DELETE FROM semantic_links WHERE subject_id = ?",
            (old_subject_id,),
        )
        for message_id, new_subject_id in assignments.items():
            conn.execute(
                """
                INSERT INTO semantic_links(
                    message_id, subject_id, confidence, created_by, updated_by
                ) VALUES (?, ?, 1.0, ?, ?)
                ON CONFLICT(message_id, subject_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    updated_date_utc = CURRENT_TIMESTAMP,
                    updated_by = excluded.updated_by
                """,
                (message_id, new_subject_id, actor, actor),
            )

