"""
Tests for subjects.py - subject lifecycle and memory files.
"""
import json
from pathlib import Path

import pytest

import subjects as _subjects


# ---------------------------------------------------------------------------
# create_subject / assign_dewey_id
# ---------------------------------------------------------------------------

class TestAssignDeweyId:
    def test_first_subject_gets_zero(self, tmp_path):
        s = _subjects.create_subject("Topic A", "", None, tmp_path)
        assert s["dewey_id"] == "0"

    def test_second_top_level_gets_one(self, tmp_path):
        _subjects.create_subject("Topic A", "", None, tmp_path)
        s2 = _subjects.create_subject("Topic B", "", None, tmp_path)
        assert s2["dewey_id"] == "1"

    def test_child_gets_parent_dot_zero(self, tmp_path):
        parent = _subjects.create_subject("Parent", "", None, tmp_path)
        child = _subjects.create_subject("Child", "", parent["id"], tmp_path)
        assert child["dewey_id"] == "0.0"

    def test_second_child_gets_parent_dot_one(self, tmp_path):
        parent = _subjects.create_subject("Parent", "", None, tmp_path)
        _subjects.create_subject("Child A", "", parent["id"], tmp_path)
        child2 = _subjects.create_subject("Child B", "", parent["id"], tmp_path)
        assert child2["dewey_id"] == "0.1"

    def test_duplicate_name_raises(self, tmp_path):
        _subjects.create_subject("Same Name", "", None, tmp_path)
        with pytest.raises(ValueError, match="already exists"):
            _subjects.create_subject("Same Name", "", None, tmp_path)


# ---------------------------------------------------------------------------
# get_subject
# ---------------------------------------------------------------------------

class TestGetSubject:
    def test_lookup_by_short_id(self, tmp_path):
        s = _subjects.create_subject("Alpha", "desc", None, tmp_path)
        found = _subjects.get_subject(s["id"], tmp_path)
        assert found is not None
        assert found["name"] == "Alpha"

    def test_lookup_by_dewey_id(self, tmp_path):
        _subjects.create_subject("Alpha", "desc", None, tmp_path)
        found = _subjects.get_subject("0", tmp_path)
        assert found is not None
        assert found["dewey_id"] == "0"

    def test_missing_returns_none(self, tmp_path):
        assert _subjects.get_subject("nonexistent", tmp_path) is None


# ---------------------------------------------------------------------------
# set_subject_state / increment_message_count
# ---------------------------------------------------------------------------

class TestStateAndCount:
    def test_set_state_dormant(self, tmp_path):
        s = _subjects.create_subject("Foo", "", None, tmp_path)
        updated = _subjects.set_subject_state(s["id"], "dormant", tmp_path)
        assert updated["state"] == "dormant"
        # Persisted
        reloaded = _subjects.get_subject(s["id"], tmp_path)
        assert reloaded["state"] == "dormant"

    def test_increment_message_count(self, tmp_path):
        s = _subjects.create_subject("Baz", "", None, tmp_path)
        _subjects.increment_message_count(s["id"], delta=3, root=tmp_path)
        reloaded = _subjects.get_subject(s["id"], tmp_path)
        assert reloaded["message_count"] == 3

    def test_missing_subject_raises(self, tmp_path):
        with pytest.raises(KeyError):
            _subjects.set_subject_state("nonexistent", "dormant", tmp_path)


# ---------------------------------------------------------------------------
# write / read subject memory
# ---------------------------------------------------------------------------

class TestSubjectMemory:
    def test_write_and_read(self, tmp_path):
        s = _subjects.create_subject("Mem Test", "", None, tmp_path)
        _subjects.write_subject_memory(s, "Dense technical summary here.", tmp_path)
        content = _subjects.read_subject_memory(s, tmp_path)
        assert content == "Dense technical summary here."

    def test_read_nonexistent_returns_empty(self, tmp_path):
        s = _subjects.create_subject("Empty", "", None, tmp_path)
        content = _subjects.read_subject_memory(s, tmp_path)
        assert content == ""

    def test_write_replaces_existing(self, tmp_path):
        s = _subjects.create_subject("Overwrite", "", None, tmp_path)
        _subjects.write_subject_memory(s, "First version.", tmp_path)
        _subjects.write_subject_memory(s, "Second version.", tmp_path)
        content = _subjects.read_subject_memory(s, tmp_path)
        assert content == "Second version."


# ---------------------------------------------------------------------------
# list_subjects_summary
# ---------------------------------------------------------------------------

class TestListSubjectsSummary:
    def test_returns_list_without_memory(self, tmp_path):
        _subjects.create_subject("A", "desc a", None, tmp_path)
        _subjects.create_subject("B", "desc b", None, tmp_path)
        summary = _subjects.list_subjects_summary(tmp_path)
        assert len(summary) == 2
        for s in summary:
            assert "id" in s
            assert "name" in s
            assert "state" in s
            assert "dewey_id" in s
            # Memory content must not be embedded in summary
            assert "memory" not in s

    def test_empty_returns_empty_list(self, tmp_path):
        assert _subjects.list_subjects_summary(tmp_path) == []
