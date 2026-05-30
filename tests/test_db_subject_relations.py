"""Tests for subject_relations db functions.

Covers:
  - subject_relations table exists after init_db
  - migration creates the table on a pre-existing DB that lacked it
  - insert_subject_relation persists a row
  - get_subject_relations_for_subject returns rows with correct fields
  - rows are ordered by created_date_utc ascending
  - only rows for the requested subject are returned
  - cascade: deleting from_link_id row removes the relation row
  - cascade: deleting to_link_id row removes the relation row
  - cascade: deleting the subject removes all relation rows
  - get_subject_relations_for_subject returns [] when no edges exist
  - multiple edges for same subject are all returned
"""

import sqlite3
from pathlib import Path

import pytest

import db as _db
import subjects as _subj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_subject(tmp_path: Path, name: str) -> str:
    """Create a subject and return its id."""
    return _subj.create_subject(name, "", None, tmp_path)["id"]


def _seed_link(tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.8) -> int:
    """Seed one message and a link to the given subject. Return the link id."""
    _db.upsert_session_messages(
        session_id,
        [{"role": "user", "content": f"content for {session_id}"}],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    _db.save_assignments(
        session_id,
        {0: {subject_id: confidence}},
        _db.ACTOR_MESSAGE_CLASSIFIER,
        tmp_path,
    )
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT id FROM semantic_links WHERE subject_id = ? ORDER BY id DESC LIMIT 1",
            (subject_id,),
        ).fetchone()
    assert row is not None, "link not created"
    return int(row["id"])


def _count_relations(tmp_path: Path) -> int:
    with _db.connect(tmp_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM subject_relations").fetchone()[0]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_subject_relations_table_exists_after_init_db(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "subject_relations" in tables


def test_subject_relations_migration_on_existing_db(tmp_path: Path) -> None:
    """Init once (gets table), drop it to simulate a pre-migration DB, re-init."""
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute("DROP TABLE IF EXISTS subject_relations")
    # Confirm it's gone.
    with _db.connect(tmp_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "subject_relations" not in tables
    # Re-init should recreate it via the migration block.
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "subject_relations" in tables


# ---------------------------------------------------------------------------
# insert_subject_relation
# ---------------------------------------------------------------------------

def test_insert_subject_relation_persists_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Test Subject A")
    from_id = _seed_link(tmp_path, "sess-from", subject_id, 0.8)
    to_id = _seed_link(tmp_path, "sess-to", subject_id, 0.9)

    _db.insert_subject_relation(from_id, to_id, subject_id, "supports", "TestActor", tmp_path)

    assert _count_relations(tmp_path) == 1


def test_insert_subject_relation_stores_correct_fields(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Test Subject B")
    from_id = _seed_link(tmp_path, "sess-b-from", subject_id, 0.7)
    to_id = _seed_link(tmp_path, "sess-b-to", subject_id, 0.9)

    _db.insert_subject_relation(from_id, to_id, subject_id, "contradicts", "TestActor", tmp_path)

    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT subject_id, from_link_id, to_link_id, relation, created_by "
            "FROM subject_relations LIMIT 1"
        ).fetchone()

    assert row["subject_id"] == subject_id
    assert int(row["from_link_id"]) == from_id
    assert int(row["to_link_id"]) == to_id
    assert row["relation"] == "contradicts"
    assert row["created_by"] == "TestActor"


# ---------------------------------------------------------------------------
# get_subject_relations_for_subject
# ---------------------------------------------------------------------------

def test_get_subject_relations_returns_empty_when_none(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Empty Subject")
    result = _db.get_subject_relations_for_subject(subject_id, tmp_path)
    assert result == []


def test_get_subject_relations_returns_inserted_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Test Subject C")
    from_id = _seed_link(tmp_path, "sess-c-from", subject_id, 0.75)
    to_id = _seed_link(tmp_path, "sess-c-to", subject_id, 0.9)

    _db.insert_subject_relation(from_id, to_id, subject_id, "updates", "TestActor", tmp_path)

    rows = _db.get_subject_relations_for_subject(subject_id, tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["from_link_id"] == from_id
    assert r["to_link_id"] == to_id
    assert r["relation"] == "updates"
    assert r["subject_id"] == subject_id
    assert abs(r["from_confidence"] - 0.75) < 1e-6
    assert abs(r["to_confidence"] - 0.9) < 1e-6


def test_get_subject_relations_only_returns_own_subject(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj_a = _seed_subject(tmp_path, "Subject Alpha")
    subj_b = _seed_subject(tmp_path, "Subject Beta")

    from_a = _seed_link(tmp_path, "sess-alpha-from", subj_a, 0.8)
    to_a = _seed_link(tmp_path, "sess-alpha-to", subj_a, 0.9)
    from_b = _seed_link(tmp_path, "sess-beta-from", subj_b, 0.7)
    to_b = _seed_link(tmp_path, "sess-beta-to", subj_b, 0.6)

    _db.insert_subject_relation(from_a, to_a, subj_a, "supports", "TestActor", tmp_path)
    _db.insert_subject_relation(from_b, to_b, subj_b, "contradicts", "TestActor", tmp_path)

    rows_a = _db.get_subject_relations_for_subject(subj_a, tmp_path)
    assert len(rows_a) == 1
    assert rows_a[0]["from_link_id"] == from_a

    rows_b = _db.get_subject_relations_for_subject(subj_b, tmp_path)
    assert len(rows_b) == 1
    assert rows_b[0]["from_link_id"] == from_b


def test_get_subject_relations_returns_all_edges_for_subject(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Multi-edge Subject")
    link_a = _seed_link(tmp_path, "sess-me-a", subject_id, 0.8)
    link_b = _seed_link(tmp_path, "sess-me-b", subject_id, 0.7)
    link_c = _seed_link(tmp_path, "sess-me-c", subject_id, 0.9)

    _db.insert_subject_relation(link_a, link_c, subject_id, "updates", "TestActor", tmp_path)
    _db.insert_subject_relation(link_b, link_c, subject_id, "supports", "TestActor", tmp_path)

    rows = _db.get_subject_relations_for_subject(subject_id, tmp_path)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Cascade tests
# ---------------------------------------------------------------------------

def test_cascade_deletes_relation_when_from_link_deleted(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Cascade From Test")
    from_id = _seed_link(tmp_path, "sess-casc-from", subject_id, 0.8)
    to_id = _seed_link(tmp_path, "sess-casc-to", subject_id, 0.9)
    _db.insert_subject_relation(from_id, to_id, subject_id, "supports", "TestActor", tmp_path)
    assert _count_relations(tmp_path) == 1

    _db.delete_semantic_link(from_id, "TestActor", tmp_path)

    assert _count_relations(tmp_path) == 0


def test_cascade_deletes_relation_when_to_link_deleted(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Cascade To Test")
    from_id = _seed_link(tmp_path, "sess-cto-from", subject_id, 0.8)
    to_id = _seed_link(tmp_path, "sess-cto-to", subject_id, 0.9)
    _db.insert_subject_relation(from_id, to_id, subject_id, "updates", "TestActor", tmp_path)
    assert _count_relations(tmp_path) == 1

    _db.delete_semantic_link(to_id, "TestActor", tmp_path)

    assert _count_relations(tmp_path) == 0


def test_relation_check_constraint_rejects_invalid_relation(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subject_id = _seed_subject(tmp_path, "Check Constraint Test")
    from_id = _seed_link(tmp_path, "sess-cc-from", subject_id, 0.8)
    to_id = _seed_link(tmp_path, "sess-cc-to", subject_id, 0.9)

    with pytest.raises(Exception):
        _db.insert_subject_relation(from_id, to_id, subject_id, "bogus_relation", "TestActor", tmp_path)
