"""Tests for daemon_reclass: _scan_split_subjects and _handle_classify_memory."""

from pathlib import Path
from unittest import mock

import db as _db
import daemon_reclass
import subjects as _subjects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_split_parent(root: Path, name: str = "Parent Topic") -> dict:
    """Create a subject and force it into the 'split' state."""
    subject = _subjects.create_subject(name, "test parent", None, root)
    _subjects.set_subject_state(subject["id"], "split", root)
    return _db.get_subject(subject["id"], root)


def _make_child(root: Path, parent_id: str, name: str) -> dict:
    """Create a child subject under parent_id."""
    return _subjects.create_subject(name, f"child of {name}", parent_id, root)


def _link_message(root: Path, subject_id: str) -> str:
    """
    Insert a minimal autobiographical_memory row and link it to subject_id.
    Returns the message id.

    Uses upsert_session_messages to satisfy the sessions FK, then links the
    resulting message row to subject_id.
    """
    import uuid

    session_id = str(uuid.uuid4())
    _db.upsert_session_messages(
        session_id,
        [{"role": "user", "content": "test content"}],
        "global",
        "test",
        root,
    )
    with _db.connect(root) as conn:
        row = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = ? AND message_index = 0",
            (session_id,),
        ).fetchone()
        msg_id = str(row["id"])
        conn.execute(
            """
            INSERT INTO semantic_links(message_id, subject_id, confidence, created_by, updated_by)
            VALUES (?, ?, 0.9, 'test', 'test')
            """,
            (msg_id, subject_id),
        )
    return msg_id


# ---------------------------------------------------------------------------
# TestScanSplitSubjects
# ---------------------------------------------------------------------------


class TestScanSplitSubjects:
    def test_empty_when_no_split_subjects(self, tmp_path: Path) -> None:
        """Scanner must return nothing when no subjects are in state='split'."""
        _subjects.create_subject("Normal Topic", "", None, tmp_path)
        result = daemon_reclass._scan_split_subjects(tmp_path)
        assert result == []

    def test_returns_split_subject_with_children(self, tmp_path: Path) -> None:
        """Scanner must return an item for each split subject that has children."""
        parent = _make_split_parent(tmp_path, "Split Parent")
        child_a = _make_child(tmp_path, parent["id"], "Child A")
        child_b = _make_child(tmp_path, parent["id"], "Child B")

        result = daemon_reclass._scan_split_subjects(tmp_path)

        assert len(result) == 1
        item = result[0]
        assert item["subject_id"] == parent["id"]
        assert set(item["payload"]["candidate_subject_ids"]) == {child_a["id"], child_b["id"]}

    def test_skips_split_subject_with_no_children(self, tmp_path: Path) -> None:
        """Scanner must skip a split subject that has no child subjects (logs warning)."""
        _make_split_parent(tmp_path, "Orphan Parent")
        result = daemon_reclass._scan_split_subjects(tmp_path)
        assert result == []

    def test_does_not_return_non_split_subjects(self, tmp_path: Path) -> None:
        """Scanner must only return state='split' subjects, not open/dormant/archived."""
        open_subj = _subjects.create_subject("Open Topic", "", None, tmp_path)
        _make_child(tmp_path, open_subj["id"], "Child of Open")

        result = daemon_reclass._scan_split_subjects(tmp_path)
        assert result == []

    def test_returns_multiple_split_subjects(self, tmp_path: Path) -> None:
        """Scanner must return one item per qualifying split subject."""
        p1 = _make_split_parent(tmp_path, "Parent One")
        _make_child(tmp_path, p1["id"], "Child 1A")
        p2 = _make_split_parent(tmp_path, "Parent Two")
        _make_child(tmp_path, p2["id"], "Child 2A")

        result = daemon_reclass._scan_split_subjects(tmp_path)
        assert len(result) == 2
        ids = {item["subject_id"] for item in result}
        assert ids == {p1["id"], p2["id"]}


# ---------------------------------------------------------------------------
# TestHandleClassifyMemory
# ---------------------------------------------------------------------------


class TestHandleClassifyMemory:
    def test_links_reattributed_to_highest_scoring_child(self, tmp_path: Path) -> None:
        """Messages should be linked to the child with the highest score."""
        parent = _make_split_parent(tmp_path, "Parent")
        child_a = _make_child(tmp_path, parent["id"], "Child Alpha")
        child_b = _make_child(tmp_path, parent["id"], "Child Beta")

        msg_id = _link_message(tmp_path, parent["id"])

        # classify_messages returns index 0 -> child_b is highest
        classify_result = {0: {child_a["id"]: 0.2, child_b["id"]: 0.9}}
        cfg = {}
        payload = {"candidate_subject_ids": [child_a["id"], child_b["id"]]}

        with mock.patch("daemon_reclass.meta_agent.classify_messages", return_value=classify_result):
            daemon_reclass._handle_classify_memory(tmp_path, cfg, parent["id"], payload)

        # Original link to parent must be gone.
        parent_msgs = _db.get_messages_linked_to_subject(parent["id"], tmp_path)
        assert parent_msgs == []

        # Message must now be linked to child_b.
        child_b_msgs = _db.get_messages_linked_to_subject(child_b["id"], tmp_path)
        assert any(m["id"] == msg_id for m in child_b_msgs)

    def test_unmatched_messages_fallback_to_first_child(self, tmp_path: Path) -> None:
        """Messages where classify returns empty scores fall back to children[0]."""
        parent = _make_split_parent(tmp_path, "Parent Fallback")
        child_a = _make_child(tmp_path, parent["id"], "Fallback Alpha")
        child_b = _make_child(tmp_path, parent["id"], "Fallback Beta")

        msg_id = _link_message(tmp_path, parent["id"])

        # Empty scores for message index 0 - should fall back to children[0] == child_a
        classify_result = {0: {}}
        cfg = {}
        payload = {"candidate_subject_ids": [child_a["id"], child_b["id"]]}

        with mock.patch("daemon_reclass.meta_agent.classify_messages", return_value=classify_result):
            daemon_reclass._handle_classify_memory(tmp_path, cfg, parent["id"], payload)

        child_a_msgs = _db.get_messages_linked_to_subject(child_a["id"], tmp_path)
        assert any(m["id"] == msg_id for m in child_a_msgs)

    def test_parent_archived_after_reattribution(self, tmp_path: Path) -> None:
        """Parent state must be 'archived' after successful re-attribution."""
        parent = _make_split_parent(tmp_path, "Parent Archive")
        child_a = _make_child(tmp_path, parent["id"], "Arch Child")

        _link_message(tmp_path, parent["id"])

        cfg = {}
        payload = {"candidate_subject_ids": [child_a["id"]]}

        with mock.patch(
            "daemon_reclass.meta_agent.classify_messages",
            return_value={0: {child_a["id"]: 0.8}},
        ):
            daemon_reclass._handle_classify_memory(tmp_path, cfg, parent["id"], payload)

        parent_after = _db.get_subject(parent["id"], tmp_path)
        assert parent_after["state"] == "archived"

    def test_empty_linked_messages_archived_immediately(self, tmp_path: Path) -> None:
        """If the parent has no linked messages, skip classify and archive directly."""
        parent = _make_split_parent(tmp_path, "Empty Parent")
        child_a = _make_child(tmp_path, parent["id"], "Empty Child")

        cfg = {}
        payload = {"candidate_subject_ids": [child_a["id"]]}

        with mock.patch("daemon_reclass.meta_agent.classify_messages") as mock_classify:
            daemon_reclass._handle_classify_memory(tmp_path, cfg, parent["id"], payload)
            mock_classify.assert_not_called()

        parent_after = _db.get_subject(parent["id"], tmp_path)
        assert parent_after["state"] == "archived"

    def test_raises_when_subject_id_is_none(self, tmp_path: Path) -> None:
        """Handler must raise ValueError when subject_id is None."""
        import pytest
        with pytest.raises(ValueError, match="subject_id"):
            daemon_reclass._handle_classify_memory(tmp_path, {}, None, {})

    def test_raises_when_no_valid_children(self, tmp_path: Path) -> None:
        """Handler must raise ValueError when candidate_subject_ids are all missing."""
        import pytest
        parent = _make_split_parent(tmp_path, "Childless Parent")
        _link_message(tmp_path, parent["id"])

        cfg = {}
        payload = {"candidate_subject_ids": ["nonexistent-id-1", "nonexistent-id-2"]}

        with pytest.raises(ValueError, match="no valid children"):
            daemon_reclass._handle_classify_memory(tmp_path, cfg, parent["id"], payload)
