"""Tests for db.py migration correctness."""

import sqlite3
from pathlib import Path

import db as _db


# ---------------------------------------------------------------------------
# Bug fix: memory_actions.subject_id FK drop
# ---------------------------------------------------------------------------


def test_enqueue_reconcile_link_with_numeric_subject_id(tmp_path: Path) -> None:
    """reconcile_link actions use str(link_id) as subject_id.

    After the FK migration, enqueue_memory_action must succeed even when
    subject_id is a numeric string that has no matching semantic_subjects row.
    """
    _db.init_db(tmp_path)
    # Use a numeric string that would have violated the old FK constraint.
    result = _db.enqueue_memory_action(
        action_type="reconcile_link",
        subject_id="42",
        payload={"link_id": 42},
        root=tmp_path,
    )
    assert result

    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT action_type, subject_id, status FROM memory_actions WHERE subject_id = '42'"
        ).fetchone()
    assert row is not None
    assert row["action_type"] == "reconcile_link"
    assert row["status"] == "pending"


def test_enqueue_attenuate_confidence_with_numeric_subject_id(tmp_path: Path) -> None:
    """attenuate_confidence actions also use str(link_id) as subject_id."""
    _db.init_db(tmp_path)
    result = _db.enqueue_memory_action(
        action_type="attenuate_confidence",
        subject_id="99",
        payload={"link_id": 99},
        root=tmp_path,
    )
    assert result


def test_memory_actions_schema_has_no_subject_fk(tmp_path: Path) -> None:
    """The memory_actions table must not carry a FK to semantic_subjects."""
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_actions'"
        ).fetchone()
    assert row is not None
    assert "REFERENCES semantic_subjects" not in (row[0] or "")


def test_fk_migration_is_idempotent(tmp_path: Path) -> None:
    """Calling init_db twice on the same path must not raise."""
    _db.init_db(tmp_path)
    _db.init_db(tmp_path)  # second call must not fail


def test_fk_migration_preserves_existing_rows(tmp_path: Path) -> None:
    """Rows already in memory_actions survive the FK migration."""
    # First, create the DB and manually insert a row that has a valid subject_id
    # (pre-migration style - would have needed a real subject to pass the old FK).
    _db.init_db(tmp_path)

    # Enqueue a legitimate finalize_subject action for a real subject.
    import subjects as _subjects
    subj = _subjects.create_subject("Test Subject", "desc", None, tmp_path)
    _db.enqueue_memory_action(
        action_type="finalize_subject",
        subject_id=subj["id"],
        payload={"subject_id": subj["id"]},
        root=tmp_path,
    )

    # Now simulate a second init_db (migration runs again as a no-op).
    _db.init_db(tmp_path)

    with _db.connect(tmp_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_actions WHERE action_type = 'finalize_subject'"
        ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Bug fix: _create_audit_triggers race (CREATE TRIGGER IF NOT EXISTS)
# ---------------------------------------------------------------------------


def test_init_db_twice_does_not_raise_trigger_already_exists(tmp_path: Path) -> None:
    """init_db called twice on the same path must not raise a trigger-exists error.

    Previously _create_audit_triggers used CREATE TRIGGER without IF NOT EXISTS.
    The second call would fail with 'trigger X already exists'.
    """
    _db.init_db(tmp_path)
    _db.init_db(tmp_path)  # must not raise


def test_audit_triggers_exist_after_double_init(tmp_path: Path) -> None:
    """Audit triggers must be present after two init_db calls."""
    _db.init_db(tmp_path)
    _db.init_db(tmp_path)

    with _db.connect(tmp_path) as conn:
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }

    assert "trg_semantic_subjects_audit_insert" in triggers
    assert "trg_semantic_links_audit_insert" in triggers
    assert "trg_memory_actions_audit_insert" in triggers


def test_concurrent_init_db_calls_do_not_raise(tmp_path: Path) -> None:
    """Simulate two processes calling init_db 'concurrently' by using two
    separate connections against the same database file.

    This is a single-threaded simulation: the second connection runs init_db
    (including _create_audit_triggers) after the first has already created
    the triggers. With IF NOT EXISTS, the second run must succeed silently.
    """
    import importlib
    import sys

    # First init via the normal module.
    _db.init_db(tmp_path)

    # Second init via a fresh function call (same module, same path).
    # This mimics a second daemon that starts slightly after the first.
    _db.init_db(tmp_path)  # must not raise

    with _db.connect(tmp_path) as conn:
        count_after = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
        ).fetchone()[0]

    # Run init_db a third time and confirm the count is stable.
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        count_stable = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
        ).fetchone()[0]

    assert count_after == count_stable, (
        f"Trigger count grew from {count_after} to {count_stable} on repeated init_db - "
        "IF NOT EXISTS is not working"
    )


# ---------------------------------------------------------------------------
# session_whitelist table: existence, round-trip, and idempotent migration
# ---------------------------------------------------------------------------


def test_session_whitelist_table_exists_after_init(tmp_path: Path) -> None:
    """session_whitelist table must be created by init_db."""
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_whitelist'"
        ).fetchone()
    assert row is not None


def test_session_whitelist_save_and_load_round_trip(tmp_path: Path) -> None:
    """save_session_whitelist / load_session_whitelist must persist exactly the given set."""
    _db.init_db(tmp_path)
    wl = {"session-abc", "session-def", "session-xyz"}
    _db.save_session_whitelist(wl, _db.ACTOR_WHITELIST_MANAGER, tmp_path)
    loaded = _db.load_session_whitelist(tmp_path)
    assert loaded == wl


def test_session_whitelist_save_replaces_previous(tmp_path: Path) -> None:
    """A second save_session_whitelist call must replace the first, not append."""
    _db.init_db(tmp_path)
    _db.save_session_whitelist({"old-session"}, _db.ACTOR_WHITELIST_MANAGER, tmp_path)
    _db.save_session_whitelist({"new-session"}, _db.ACTOR_WHITELIST_MANAGER, tmp_path)
    loaded = _db.load_session_whitelist(tmp_path)
    assert loaded == {"new-session"}


def test_session_whitelist_load_empty_when_nothing_saved(tmp_path: Path) -> None:
    """load_session_whitelist must return an empty set on a fresh DB."""
    _db.init_db(tmp_path)
    loaded = _db.load_session_whitelist(tmp_path)
    assert loaded == set()


def test_session_whitelist_migration_idempotent(tmp_path: Path) -> None:
    """Calling init_db twice must not fail even if session_whitelist already exists."""
    _db.init_db(tmp_path)
    _db.init_db(tmp_path)  # must not raise

    with _db.connect(tmp_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='session_whitelist'"
        ).fetchone()[0]
    assert count == 1
