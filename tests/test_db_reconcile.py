"""Tests for reconcile-related db functions.

Covers:
  - reconciled_at column exists after init_db
  - get_links_pending_reconciliation returns only unreconciled links
  - mark_link_reconciled stamps a timestamp
  - mark_link_reconciled is idempotent
"""

from pathlib import Path

import pytest

import db as _db
import subjects as _subj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_link(tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.9) -> int:
    """Seed one message and link it to a subject. Return the semantic_links.id."""
    _db.upsert_session_messages(
        session_id,
        [{"role": "user", "content": "test content for reconcile"}],
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
            "SELECT id FROM semantic_links WHERE subject_id = ? LIMIT 1",
            (subject_id,),
        ).fetchone()
    assert row is not None, "link not created"
    return int(row["id"])


def _get_reconciled_at(tmp_path: Path, link_id: int) -> str | None:
    """Return reconciled_at for a given link_id."""
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT reconciled_at FROM semantic_links WHERE id = ?",
            (link_id,),
        ).fetchone()
    assert row is not None
    return row["reconciled_at"]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_reconciled_at_column_exists_after_init_db(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    with _db.connect(tmp_path) as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(semantic_links)")}
    assert "reconciled_at" in cols


def test_reconciled_at_starts_null_for_new_links(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Reconcile Test", "test subject", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-null-1", subj["id"])
    assert _get_reconciled_at(tmp_path, link_id) is None


# ---------------------------------------------------------------------------
# get_links_pending_reconciliation
# ---------------------------------------------------------------------------

def test_get_links_pending_reconciliation_returns_unreconciled_only(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj_a = _subj.create_subject("Subject A", "desc", None, tmp_path)
    subj_b = _subj.create_subject("Subject B", "desc", None, tmp_path)

    link_a = _seed_link(tmp_path, "sess-rec-a", subj_a["id"])
    link_b = _seed_link(tmp_path, "sess-rec-b", subj_b["id"])

    # Mark link_b reconciled manually
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_links SET reconciled_at = CURRENT_TIMESTAMP WHERE id = ?",
            (link_b,),
        )

    pending = _db.get_links_pending_reconciliation(tmp_path)
    pending_ids = {p["id"] for p in pending}

    assert link_a in pending_ids
    assert link_b not in pending_ids


def test_get_links_pending_reconciliation_empty_when_all_reconciled(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("All Reconciled", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-all-rec", subj["id"])

    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_links SET reconciled_at = CURRENT_TIMESTAMP WHERE id = ?",
            (link_id,),
        )

    pending = _db.get_links_pending_reconciliation(tmp_path)
    assert pending == []


def test_get_links_pending_reconciliation_empty_db(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    assert _db.get_links_pending_reconciliation(tmp_path) == []


def test_get_links_pending_reconciliation_returns_required_fields(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Fields Check", "desc", None, tmp_path)
    _seed_link(tmp_path, "sess-fields", subj["id"])

    pending = _db.get_links_pending_reconciliation(tmp_path)
    assert len(pending) >= 1
    row = pending[0]
    for key in ("id", "message_id", "subject_id", "confidence", "created_date_utc"):
        assert key in row, f"missing key: {key}"


def test_get_links_pending_reconciliation_oldest_first(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj_a = _subj.create_subject("Oldest First A", "desc", None, tmp_path)
    subj_b = _subj.create_subject("Oldest First B", "desc", None, tmp_path)

    link_a = _seed_link(tmp_path, "sess-ord-a", subj_a["id"])
    link_b = _seed_link(tmp_path, "sess-ord-b", subj_b["id"])

    pending = _db.get_links_pending_reconciliation(tmp_path)
    pending_ids = [p["id"] for p in pending]
    # link_a was created first so it should appear before link_b
    assert pending_ids.index(link_a) < pending_ids.index(link_b)


# ---------------------------------------------------------------------------
# mark_link_reconciled
# ---------------------------------------------------------------------------

def test_mark_link_reconciled_sets_timestamp(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Mark Reconciled", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-mark-1", subj["id"])

    assert _get_reconciled_at(tmp_path, link_id) is None
    _db.mark_link_reconciled(link_id, _db.ACTOR_MEMORY_ACTIONS, tmp_path)
    assert _get_reconciled_at(tmp_path, link_id) is not None


def test_mark_link_reconciled_removes_from_pending(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Remove Pending", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-rem-pend", subj["id"])

    assert any(p["id"] == link_id for p in _db.get_links_pending_reconciliation(tmp_path))
    _db.mark_link_reconciled(link_id, _db.ACTOR_MEMORY_ACTIONS, tmp_path)
    assert not any(p["id"] == link_id for p in _db.get_links_pending_reconciliation(tmp_path))


def test_mark_link_reconciled_idempotent(tmp_path: Path) -> None:
    """Calling mark_link_reconciled twice must not raise."""
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Idempotent", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-idem", subj["id"])

    _db.mark_link_reconciled(link_id, _db.ACTOR_MEMORY_ACTIONS, tmp_path)
    first_ts = _get_reconciled_at(tmp_path, link_id)
    _db.mark_link_reconciled(link_id, _db.ACTOR_MEMORY_ACTIONS, tmp_path)  # second call
    # Should not raise; timestamp may have updated (that is acceptable)
    assert _get_reconciled_at(tmp_path, link_id) is not None
    assert first_ts is not None


def test_mark_link_reconciled_updates_updated_by(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Updated By", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-updby", subj["id"])

    actor = "TestReconciler"
    _db.mark_link_reconciled(link_id, actor, tmp_path)

    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT updated_by FROM semantic_links WHERE id = ?",
            (link_id,),
        ).fetchone()
    assert row["updated_by"] == actor
