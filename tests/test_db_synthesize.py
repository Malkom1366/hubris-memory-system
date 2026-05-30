"""Tests for the new synthesize-related DB functions.

Covers:
  - synthesized_at column exists after init_db
  - get_subjects_needing_split: returns over-threshold non-archived subjects
  - get_subjects_needing_split: excludes archived subjects
  - get_subjects_needing_split: excludes subjects below threshold
  - get_archived_subjects_needing_synthesis: returns archived subjects with NULL synthesized_at
  - get_archived_subjects_needing_synthesis: excludes already-synthesized subjects
  - get_archived_subjects_needing_synthesis: excludes non-archived subjects
  - mark_subject_synthesized: stamps synthesized_at, excludes subject from future queries
"""

from pathlib import Path

import db as _db
import subjects as _subjects


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSynthesizedAtColumn:
    def test_column_exists_after_init_db(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        with _db.connect(tmp_path) as conn:
            cols = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(semantic_subjects)"
                ).fetchall()
            }
        assert "synthesized_at" in cols


# ---------------------------------------------------------------------------
# get_subjects_needing_split
# ---------------------------------------------------------------------------


class TestGetSubjectsNeedingSplit:
    def test_returns_over_threshold_subjects(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Fat Subject", "", None, tmp_path)
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
                (subj["id"],),
            )
        rows = _db.get_subjects_needing_split(150, tmp_path)
        assert any(r["id"] == subj["id"] for r in rows)

    def test_excludes_below_threshold(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Small Subject", "", None, tmp_path)
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "UPDATE semantic_subjects SET message_count = 10 WHERE id = ?",
                (subj["id"],),
            )
        rows = _db.get_subjects_needing_split(150, tmp_path)
        assert not any(r["id"] == subj["id"] for r in rows)

    def test_excludes_archived_subjects(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Archived Fat", "", None, tmp_path)
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "UPDATE semantic_subjects SET message_count = 200, state = 'archived' WHERE id = ?",
                (subj["id"],),
            )
        rows = _db.get_subjects_needing_split(150, tmp_path)
        assert not any(r["id"] == subj["id"] for r in rows)

    def test_result_contains_expected_fields(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Fields Check", "", None, tmp_path)
        with _db.connect(tmp_path) as conn:
            conn.execute(
                "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
                (subj["id"],),
            )
        rows = _db.get_subjects_needing_split(150, tmp_path)
        matching = [r for r in rows if r["id"] == subj["id"]]
        assert len(matching) == 1
        row = matching[0]
        assert "id" in row
        assert "name" in row
        assert "message_count" in row
        assert row["message_count"] >= 150

    def test_returns_empty_when_no_subjects_qualify(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        assert _db.get_subjects_needing_split(150, tmp_path) == []


# ---------------------------------------------------------------------------
# get_archived_subjects_needing_synthesis
# ---------------------------------------------------------------------------


class TestGetArchivedSubjectsNeedingSynthesis:
    def test_returns_archived_with_null_synthesized_at(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Needs Synthesis", "", None, tmp_path)
        _subjects.set_subject_state(subj["id"], "archived", tmp_path)
        rows = _db.get_archived_subjects_needing_synthesis(tmp_path)
        assert any(r["id"] == subj["id"] for r in rows)

    def test_excludes_already_synthesized(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Already Done", "", None, tmp_path)
        _subjects.set_subject_state(subj["id"], "archived", tmp_path)
        _db.mark_subject_synthesized(subj["id"], _db.ACTOR_MEMORY_ACTIONS, tmp_path)
        rows = _db.get_archived_subjects_needing_synthesis(tmp_path)
        assert not any(r["id"] == subj["id"] for r in rows)

    def test_excludes_non_archived_subjects(self, tmp_path: Path) -> None:
        open_subj = _subjects.create_subject("Open One", "", None, tmp_path)
        dormant_subj = _subjects.create_subject("Dormant One", "", None, tmp_path)
        _subjects.set_subject_state(dormant_subj["id"], "dormant", tmp_path)
        rows = _db.get_archived_subjects_needing_synthesis(tmp_path)
        ids = {r["id"] for r in rows}
        assert open_subj["id"] not in ids
        assert dormant_subj["id"] not in ids

    def test_result_contains_expected_fields(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Fields Only", "", None, tmp_path)
        _subjects.set_subject_state(subj["id"], "archived", tmp_path)
        rows = _db.get_archived_subjects_needing_synthesis(tmp_path)
        matching = [r for r in rows if r["id"] == subj["id"]]
        assert len(matching) == 1
        assert "id" in matching[0]
        assert "name" in matching[0]

    def test_returns_empty_when_nothing_archived(self, tmp_path: Path) -> None:
        _db.init_db(tmp_path)
        assert _db.get_archived_subjects_needing_synthesis(tmp_path) == []


# ---------------------------------------------------------------------------
# mark_subject_synthesized
# ---------------------------------------------------------------------------


class TestMarkSubjectSynthesized:
    def test_stamps_synthesized_at(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Mark Me", "", None, tmp_path)
        _subjects.set_subject_state(subj["id"], "archived", tmp_path)
        _db.mark_subject_synthesized(subj["id"], _db.ACTOR_MEMORY_ACTIONS, tmp_path)
        with _db.connect(tmp_path) as conn:
            row = conn.execute(
                "SELECT synthesized_at FROM semantic_subjects WHERE id = ?",
                (subj["id"],),
            ).fetchone()
        assert row is not None
        assert row["synthesized_at"] is not None

    def test_idempotent_on_repeated_calls(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("Mark Twice", "", None, tmp_path)
        _subjects.set_subject_state(subj["id"], "archived", tmp_path)
        _db.mark_subject_synthesized(subj["id"], _db.ACTOR_MEMORY_ACTIONS, tmp_path)
        _db.mark_subject_synthesized(subj["id"], _db.ACTOR_MEMORY_ACTIONS, tmp_path)
        rows = _db.get_archived_subjects_needing_synthesis(tmp_path)
        assert not any(r["id"] == subj["id"] for r in rows)

    def test_removes_subject_from_needing_synthesis_query(self, tmp_path: Path) -> None:
        subj = _subjects.create_subject("No Longer Needed", "", None, tmp_path)
        _subjects.set_subject_state(subj["id"], "archived", tmp_path)

        before = _db.get_archived_subjects_needing_synthesis(tmp_path)
        assert any(r["id"] == subj["id"] for r in before)

        _db.mark_subject_synthesized(subj["id"], _db.ACTOR_MEMORY_ACTIONS, tmp_path)

        after = _db.get_archived_subjects_needing_synthesis(tmp_path)
        assert not any(r["id"] == subj["id"] for r in after)
