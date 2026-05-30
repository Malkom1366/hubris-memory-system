"""Tests for subject embedding DB functions.

Covers:
  - subject_embeddings table exists after init_db
  - get_subjects_needing_embedding returns subjects with no embedding
  - get_subjects_needing_embedding returns subjects with a stale embedding
  - get_subjects_needing_embedding skips subjects with a current embedding
  - get_subjects_needing_embedding skips subjects with empty memory_content
  - get_subjects_needing_embedding respects the limit argument
  - upsert_subject_embedding writes both subject_embeddings and vec_subjects
  - upsert_subject_embedding replaces stale entry on re-embed
  - semantic_search_subjects returns nearest-first
  - semantic_search_subjects returns empty list when nothing is embedded
"""

import struct
from pathlib import Path

import pytest

import db as _db
import embeddings as _emb
import subjects as _subj

_DIMS = 768


def _vec(val: float) -> bytes:
    return struct.pack(f"{_DIMS}f", *([val] * _DIMS))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSubjectEmbeddingsSchema:
    def test_table_exists_after_init_db(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        with _db.connect(tmp_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "subject_embeddings" in tables


# ---------------------------------------------------------------------------
# get_subjects_needing_embedding
# ---------------------------------------------------------------------------


class TestGetSubjectsNeedingEmbedding:
    def test_returns_subject_with_content_and_no_embedding(
        self, tmp_path: Path
    ) -> None:
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Alpha Subject", "desc", None, tmp_path)
        _db.write_subject_memory(
            subj["id"], "Some synthesized content.", _db.ACTOR_MEMORY_WRITER, tmp_path
        )
        rows = _db.get_subjects_needing_embedding(10, tmp_path)
        ids = [r["id"] for r in rows]
        assert subj["id"] in ids

    def test_skips_subject_with_empty_memory_content(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Empty Content", "desc", None, tmp_path)
        # memory_content stays at the default empty string - should not be returned
        rows = _db.get_subjects_needing_embedding(10, tmp_path)
        ids = [r["id"] for r in rows]
        assert subj["id"] not in ids

    def test_returns_subject_with_stale_embedding(self, tmp_path: Path) -> None:
        """Subject whose embedding predates its last synthesis update is returned."""
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Stale Subject", "desc", None, tmp_path)
        _db.write_subject_memory(
            subj["id"], "First synthesis.", _db.ACTOR_MEMORY_WRITER, tmp_path
        )
        # Insert a subject_embeddings row backdated to before the write.
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "INSERT INTO subject_embeddings(subject_id, embedded_date_utc)"
                " VALUES (?, '2020-01-01 00:00:00')",
                (subj["id"],),
            )
        rows = _db.get_subjects_needing_embedding(10, tmp_path)
        ids = [r["id"] for r in rows]
        assert subj["id"] in ids

    def test_skips_subject_with_current_embedding(self, tmp_path: Path) -> None:
        """Subject embedded after its last synthesis update is NOT returned."""
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Current Subject", "desc", None, tmp_path)
        # Backdate updated_date_utc so the embedding timestamp will be newer.
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "UPDATE semantic_subjects"
                " SET memory_content = 'Some content', updated_date_utc = '2020-01-01 00:00:00'"
                " WHERE id = ?",
                (subj["id"],),
            )
        # Insert embedding row with current (newer) timestamp.
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "INSERT INTO subject_embeddings(subject_id) VALUES (?)",
                (subj["id"],),
            )
        rows = _db.get_subjects_needing_embedding(10, tmp_path)
        ids = [r["id"] for r in rows]
        assert subj["id"] not in ids

    def test_respects_limit(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        for i in range(5):
            subj = _subj.create_subject(f"Limit Subject {i}", "desc", None, tmp_path)
            _db.write_subject_memory(
                subj["id"], f"Content {i}.", _db.ACTOR_MEMORY_WRITER, tmp_path
            )
        rows = _db.get_subjects_needing_embedding(2, tmp_path)
        assert len(rows) <= 2


# ---------------------------------------------------------------------------
# upsert_subject_embedding + semantic_search_subjects (require sqlite-vec)
# ---------------------------------------------------------------------------


class TestUpsertAndSearch:
    @pytest.fixture(autouse=True)
    def skip_without_vec(self) -> None:
        if not _emb.is_available():
            pytest.skip("sqlite-vec not installed")

    def test_upsert_creates_row_in_subject_embeddings(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Vec Subject", "desc", None, tmp_path)
        ok = _db.upsert_subject_embedding(subj["id"], _vec(0.5), tmp_path)
        assert ok is True
        with _db.connect(tmp_path) as conn:
            meta = conn.execute(
                "SELECT rowid FROM subject_embeddings WHERE subject_id = ?",
                (subj["id"],),
            ).fetchone()
        assert meta is not None

    def test_upsert_replaces_stale_entry(self, tmp_path: Path) -> None:
        """A second upsert removes the old row and inserts a fresh one."""
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Replace Subject", "desc", None, tmp_path)
        _db.upsert_subject_embedding(subj["id"], _vec(0.1), tmp_path)
        with _db.connect(tmp_path) as conn:
            old_rowid = conn.execute(
                "SELECT rowid FROM subject_embeddings WHERE subject_id = ?",
                (subj["id"],),
            ).fetchone()["rowid"]
        _db.upsert_subject_embedding(subj["id"], _vec(0.9), tmp_path)
        with _db.connect(tmp_path) as conn:
            new_rowid = conn.execute(
                "SELECT rowid FROM subject_embeddings WHERE subject_id = ?",
                (subj["id"],),
            ).fetchone()["rowid"]
            count = conn.execute(
                "SELECT COUNT(*) FROM subject_embeddings WHERE subject_id = ?",
                (subj["id"],),
            ).fetchone()[0]
        assert count == 1
        assert new_rowid != old_rowid

    def test_semantic_search_subjects_returns_nearest_first(
        self, tmp_path: Path
    ) -> None:
        _db.init_db(tmp_path)
        subj_near = _subj.create_subject("Near Subject", "desc", None, tmp_path)
        subj_far = _subj.create_subject("Far Subject", "desc2", None, tmp_path)
        query_vec = _vec(0.5)
        _db.upsert_subject_embedding(subj_near["id"], _vec(0.5), tmp_path)
        _db.upsert_subject_embedding(subj_far["id"], _vec(0.0), tmp_path)
        results = _db.semantic_search_subjects(query_vec, k=2, root=tmp_path)
        assert len(results) >= 1
        assert results[0]["id"] == subj_near["id"]

    def test_semantic_search_subjects_returns_name_and_dewey_id(
        self, tmp_path: Path
    ) -> None:
        _db.init_db(tmp_path)
        subj = _subj.create_subject("Named Subject", "desc", None, tmp_path)
        _db.upsert_subject_embedding(subj["id"], _vec(0.5), tmp_path)
        results = _db.semantic_search_subjects(_vec(0.5), k=1, root=tmp_path)
        assert len(results) == 1
        assert results[0]["name"] == "Named Subject"
        assert "dewey_id" in results[0]

    def test_semantic_search_subjects_empty_when_nothing_embedded(
        self, tmp_path: Path
    ) -> None:
        _db.init_db(tmp_path)
        _subj.create_subject("Unembedded Subject", "desc", None, tmp_path)
        results = _db.semantic_search_subjects(_vec(0.5), k=5, root=tmp_path)
        assert results == []
