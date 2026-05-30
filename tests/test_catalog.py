"""
Tests for catalog.py - catalog management and anchor injection.
"""
import json
from pathlib import Path

import pytest

import catalog
import subjects as _subjects
from catalog import ANCHOR_MARKER


def _write_session(path: Path, turns: list[dict]) -> None:
    path.write_text(json.dumps(turns), encoding="utf-8")


def _read_session(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# load / save round-trip
# ---------------------------------------------------------------------------

class TestCatalogRoundTrip:
    def test_load_save_round_trip(self, tmp_path):
        cat = {"version": 1, "subjects": [{"id": "abc", "name": "Foo", "dewey_id": "0"}]}
        catalog.save_catalog(cat, tmp_path)
        loaded = catalog.load_catalog(tmp_path)
        assert loaded == cat

    def test_load_missing_returns_empty(self, tmp_path):
        loaded = catalog.load_catalog(tmp_path)
        assert loaded["subjects"] == []
        assert loaded["version"] == 1


# ---------------------------------------------------------------------------
# render_anchor_text
# ---------------------------------------------------------------------------

class TestRenderAnchorText:
    def test_starts_with_marker(self, tmp_path):
        _subjects.create_subject("Topic A", "desc", None, tmp_path)
        all_subjects = _subjects.load_subjects(tmp_path)
        cat = catalog.rebuild_catalog_from_subjects(all_subjects, tmp_path)
        text = catalog.render_anchor_text(cat)
        assert text.lstrip().startswith(ANCHOR_MARKER)

    def test_contains_subject_name(self, tmp_path):
        _subjects.create_subject("UniqueSubjectName", "desc", None, tmp_path)
        all_subjects = _subjects.load_subjects(tmp_path)
        cat = catalog.rebuild_catalog_from_subjects(all_subjects, tmp_path)
        text = catalog.render_anchor_text(cat)
        assert "UniqueSubjectName" in text

    def test_empty_catalog_still_starts_with_marker(self, tmp_path):
        text = catalog.render_anchor_text({})
        assert text.lstrip().startswith(ANCHOR_MARKER)


# ---------------------------------------------------------------------------
# check_anchor
# ---------------------------------------------------------------------------

class TestCheckAnchor:
    def test_returns_true_when_anchor_present(self, tmp_path):
        session_path = tmp_path / "s.json"
        anchor_turn = {
            "message": {"role": "user", "content": f"{ANCHOR_MARKER}\nsome content", "id": "hubris-catalog-anchor"},
            "contextItems": [],
        }
        _write_session(session_path, [anchor_turn])
        assert catalog.check_anchor(session_path) is True

    def test_returns_false_when_anchor_absent(self, tmp_path):
        session_path = tmp_path / "s.json"
        normal_turn = {"message": {"role": "user", "content": "Hello", "id": "x"}, "contextItems": []}
        _write_session(session_path, [normal_turn])
        assert catalog.check_anchor(session_path) is False

    def test_returns_false_for_empty_session(self, tmp_path):
        session_path = tmp_path / "s.json"
        _write_session(session_path, [])
        assert catalog.check_anchor(session_path) is False

    def test_returns_false_for_missing_file(self, tmp_path):
        assert catalog.check_anchor(tmp_path / "nonexistent.json") is False


# ---------------------------------------------------------------------------
# inject_anchor
# ---------------------------------------------------------------------------

class TestInjectAnchor:
    def test_inserts_anchor_at_position_zero(self, tmp_path):
        session_path = tmp_path / "s.json"
        normal = {"message": {"role": "user", "content": "Hello", "id": "x"}, "contextItems": []}
        _write_session(session_path, [normal])

        _subjects.create_subject("Topic A", "", None, tmp_path)
        all_subjects = _subjects.load_subjects(tmp_path)
        cat = catalog.rebuild_catalog_from_subjects(all_subjects, tmp_path)
        catalog.inject_anchor(session_path, cat, replace_existing=False)

        turns = _read_session(session_path)
        assert turns[0]["message"]["content"].lstrip().startswith(ANCHOR_MARKER)
        assert turns[1]["message"]["content"] == "Hello"

    def test_replaces_existing_anchor(self, tmp_path):
        session_path = tmp_path / "s.json"
        old_anchor = {
            "message": {"role": "user", "content": f"{ANCHOR_MARKER}\nold content", "id": "hubris-catalog-anchor"},
            "contextItems": [],
        }
        normal = {"message": {"role": "user", "content": "Work message", "id": "x"}, "contextItems": []}
        _write_session(session_path, [old_anchor, normal])

        _subjects.create_subject("New Topic", "", None, tmp_path)
        all_subjects = _subjects.load_subjects(tmp_path)
        cat = catalog.rebuild_catalog_from_subjects(all_subjects, tmp_path)
        catalog.inject_anchor(session_path, cat, replace_existing=True)

        turns = _read_session(session_path)
        assert len(turns) == 2  # old anchor replaced, not doubled
        assert "New Topic" in turns[0]["message"]["content"]
        assert turns[1]["message"]["content"] == "Work message"


# ---------------------------------------------------------------------------
# restore_anchor_after_truncation
# ---------------------------------------------------------------------------

class TestRestoreAnchorAfterTruncation:
    def test_catalog_at_index_zero(self, tmp_path):
        session_path = tmp_path / "s.json"
        remaining = {"message": {"role": "assistant", "content": "Something that survived", "id": "y"}, "contextItems": []}
        _write_session(session_path, [remaining])

        _subjects.create_subject("Surviving Topic", "", None, tmp_path)
        all_subjects = _subjects.load_subjects(tmp_path)
        cat = catalog.rebuild_catalog_from_subjects(all_subjects, tmp_path)
        catalog.restore_anchor_after_truncation(session_path, cat, summary_text="Dropped content summary.")

        turns = _read_session(session_path)
        assert turns[0]["message"]["content"].lstrip().startswith(ANCHOR_MARKER)

    def test_summary_at_index_one_when_provided(self, tmp_path):
        session_path = tmp_path / "s.json"
        _write_session(session_path, [])

        cat = {}
        catalog.restore_anchor_after_truncation(session_path, cat, summary_text="Compacted dropped turns.")

        turns = _read_session(session_path)
        assert len(turns) == 2
        assert "[HUBRIS-TRUNCATION-RECOVERY]" in turns[1]["message"]["content"]
        assert "Compacted dropped turns." in turns[1]["message"]["content"]

    def test_no_summary_inserts_only_anchor(self, tmp_path):
        session_path = tmp_path / "s.json"
        _write_session(session_path, [])

        cat = {}
        catalog.restore_anchor_after_truncation(session_path, cat, summary_text=None)

        turns = _read_session(session_path)
        assert len(turns) == 1
        assert turns[0]["message"]["content"].lstrip().startswith(ANCHOR_MARKER)
