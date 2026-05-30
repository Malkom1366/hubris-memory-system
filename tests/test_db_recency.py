"""Tests for recency-related db functions:
  - semantic_search_messages returns link_created_date column
  - semantic_search_subject_messages scopes to a single subject
"""

import struct
from pathlib import Path

import pytest

import db as _db
import embeddings as _emb
import subjects as _subjects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIMS = 768
_ZERO_VEC = struct.pack(f"{_DIMS}f", *([0.0] * _DIMS))


def _require_vec() -> None:
    if not _emb.is_available():
        pytest.skip("sqlite-vec not installed")


def _seed_message(tmp_path: Path, session_id: str, content: str = "test message") -> dict:
    """Insert one message and return a dict with id and rowid."""
    _db.upsert_session_messages(
        session_id,
        [{"role": "user", "content": content}],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT id, rowid FROM autobiographical_memory WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
    return dict(row)


def _embed_message(tmp_path: Path, rowid: int, vec: bytes) -> None:
    _db.upsert_message_embeddings_batch({rowid: vec}, tmp_path)


def _link_message(
    tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.9
) -> None:
    _db.save_assignments(
        session_id,
        {0: {subject_id: confidence}},
        _db.ACTOR_MESSAGE_CLASSIFIER,
        tmp_path,
    )


# ---------------------------------------------------------------------------
# semantic_search_messages - link_created_date column
# ---------------------------------------------------------------------------


def test_semantic_search_messages_includes_link_created_date_key(tmp_path: Path) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    subject = _subjects.create_subject("Render Jobs", "desc", None, tmp_path)
    row = _seed_message(tmp_path, "sess-date-1", "render job alpha")
    _embed_message(tmp_path, row["rowid"], _ZERO_VEC)
    _link_message(tmp_path, "sess-date-1", subject["id"])

    results = _db.semantic_search_messages(_ZERO_VEC, k=5, root=tmp_path)

    assert len(results) >= 1
    for r in results:
        assert "link_created_date" in r


def test_link_created_date_is_non_none_for_linked_message(tmp_path: Path) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    subject = _subjects.create_subject("Render Jobs", "desc", None, tmp_path)
    row = _seed_message(tmp_path, "sess-date-2", "render job beta")
    _embed_message(tmp_path, row["rowid"], _ZERO_VEC)
    _link_message(tmp_path, "sess-date-2", subject["id"])

    results = _db.semantic_search_messages(_ZERO_VEC, k=5, root=tmp_path)
    linked = [r for r in results if r["session_id"] == "sess-date-2"]

    assert linked, "seeded message not found in results"
    assert linked[0]["link_created_date"] is not None


def test_link_created_date_is_none_for_unlinked_message(tmp_path: Path) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    row = _seed_message(tmp_path, "sess-date-3", "orphan message no subject")
    _embed_message(tmp_path, row["rowid"], _ZERO_VEC)
    # No _link_message call - this message has no semantic link.

    results = _db.semantic_search_messages(_ZERO_VEC, k=5, root=tmp_path)
    orphan = [r for r in results if r["session_id"] == "sess-date-3"]

    assert orphan, "seeded message not found in results"
    assert orphan[0]["link_created_date"] is None


# ---------------------------------------------------------------------------
# semantic_search_subject_messages - subject-scoped KNN
# ---------------------------------------------------------------------------


def test_semantic_search_subject_messages_returns_linked_messages(tmp_path: Path) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    subject = _subjects.create_subject("Topic A", "desc", None, tmp_path)
    row = _seed_message(tmp_path, "sess-subj-1", "message about topic a")
    _embed_message(tmp_path, row["rowid"], _ZERO_VEC)
    _link_message(tmp_path, "sess-subj-1", subject["id"])

    results = _db.semantic_search_subject_messages(_ZERO_VEC, subject["id"], k=5, root=tmp_path)

    assert len(results) >= 1
    assert results[0]["session_id"] == "sess-subj-1"


def test_semantic_search_subject_messages_excludes_other_subject(tmp_path: Path) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    subject_a = _subjects.create_subject("Topic A", "desc", None, tmp_path)
    subject_b = _subjects.create_subject("Topic B", "desc", None, tmp_path)
    row_a = _seed_message(tmp_path, "sess-subj-2", "topic a content")
    row_b = _seed_message(tmp_path, "sess-subj-3", "topic b content")
    _embed_message(tmp_path, row_a["rowid"], _ZERO_VEC)
    _embed_message(tmp_path, row_b["rowid"], _ZERO_VEC)
    _link_message(tmp_path, "sess-subj-2", subject_a["id"])
    _link_message(tmp_path, "sess-subj-3", subject_b["id"])

    results = _db.semantic_search_subject_messages(
        _ZERO_VEC, subject_a["id"], k=10, root=tmp_path
    )
    session_ids = {r["session_id"] for r in results}

    assert "sess-subj-2" in session_ids
    assert "sess-subj-3" not in session_ids


def test_semantic_search_subject_messages_result_has_link_fields(tmp_path: Path) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    subject = _subjects.create_subject("Topic A", "desc", None, tmp_path)
    row = _seed_message(tmp_path, "sess-subj-4", "another topic a message")
    _embed_message(tmp_path, row["rowid"], _ZERO_VEC)
    _link_message(tmp_path, "sess-subj-4", subject["id"])

    results = _db.semantic_search_subject_messages(
        _ZERO_VEC, subject["id"], k=5, root=tmp_path
    )

    assert len(results) >= 1
    r = results[0]
    assert "link_id" in r
    assert "link_created_date" in r
    assert "link_confidence" in r
    assert r["link_created_date"] is not None
    assert r["link_confidence"] > 0.0


def test_semantic_search_subject_messages_empty_for_subject_with_no_links(
    tmp_path: Path,
) -> None:
    _require_vec()
    _db.init_db(tmp_path)
    subject = _subjects.create_subject("Empty Subject", "desc", None, tmp_path)

    results = _db.semantic_search_subject_messages(
        _ZERO_VEC, subject["id"], k=5, root=tmp_path
    )

    assert results == []
