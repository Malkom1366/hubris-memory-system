"""Tests for the recall_subject server tool.

Covers:
  - subject not found returns error string
  - subject with no memory content returns 'no memory content' message
  - subject with memory but no relation edges returns content only
  - subject with relation edges appends 'Belief history' section
  - belief history section lists all edges with correct relation labels
  - belief history section is omitted gracefully when db query raises
"""

from pathlib import Path
from unittest import mock

import pytest

import db as _db
import subjects as _subj
import server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path) -> dict:
    return {"workspace_id": "test", "max_recall_tokens": 6000}


def _patch_root(tmp_path: Path):
    """Context manager: patch config so recall_subject uses tmp_path as root."""
    return mock.patch("server._config.memory_root", return_value=tmp_path)


def _patch_cfg(tmp_path: Path):
    return mock.patch("server._config.load", return_value=_make_cfg(tmp_path))


def _seed_subject(tmp_path: Path, name: str) -> dict:
    return _subj.create_subject(name, "", None, tmp_path)


def _seed_link(tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.8) -> int:
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
    assert row is not None
    return int(row["id"])


# ---------------------------------------------------------------------------
# Subject not found / no memory
# ---------------------------------------------------------------------------

def test_recall_subject_unknown_id_returns_error(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    with _patch_cfg(tmp_path), _patch_root(tmp_path):
        result = server.recall_subject(id="does-not-exist")
    assert "not found" in result.lower()


def test_recall_subject_no_memory_content_returns_message(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _seed_subject(tmp_path, "Empty Subject RS")
    with _patch_cfg(tmp_path), _patch_root(tmp_path):
        # read_subject_memory returns empty string for a newly created subject
        with mock.patch("server._subjects.read_subject_memory", return_value=""):
            result = server.recall_subject(id=subj["id"])
    assert "no memory content" in result.lower()


# ---------------------------------------------------------------------------
# No relation edges
# ---------------------------------------------------------------------------

def test_recall_subject_no_edges_returns_content_only(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _seed_subject(tmp_path, "No Edges Subject")
    _seed_link(tmp_path, "sess-ne", subj["id"])

    with _patch_cfg(tmp_path), _patch_root(tmp_path):
        with mock.patch("server._subjects.read_subject_memory", return_value="Some memory text."):
            with mock.patch("server._subjects.parse_memory_view", return_value="Some memory text."):
                result = server.recall_subject(id=subj["id"])

    assert "Some memory text." in result
    assert "Belief history" not in result


# ---------------------------------------------------------------------------
# With relation edges
# ---------------------------------------------------------------------------

def test_recall_subject_with_edges_shows_belief_history(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _seed_subject(tmp_path, "Edges Subject")
    from_id = _seed_link(tmp_path, "sess-e-from", subj["id"], confidence=0.75)
    to_id = _seed_link(tmp_path, "sess-e-to", subj["id"], confidence=0.9)
    _db.insert_subject_relation(from_id, to_id, subj["id"], "updates", "TestActor", tmp_path)

    with _patch_cfg(tmp_path), _patch_root(tmp_path):
        with mock.patch("server._subjects.read_subject_memory", return_value="Memory text."):
            with mock.patch("server._subjects.parse_memory_view", return_value="Memory text."):
                result = server.recall_subject(id=subj["id"])

    assert "Belief history" in result
    assert "--updates-->" in result
    assert str(from_id) in result
    assert str(to_id) in result


def test_recall_subject_belief_history_lists_all_edges(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _seed_subject(tmp_path, "Multi Edges Subject")
    link_a = _seed_link(tmp_path, "sess-ma", subj["id"], confidence=0.8)
    link_b = _seed_link(tmp_path, "sess-mb", subj["id"], confidence=0.7)
    link_c = _seed_link(tmp_path, "sess-mc", subj["id"], confidence=0.9)
    _db.insert_subject_relation(link_a, link_c, subj["id"], "updates", "TestActor", tmp_path)
    _db.insert_subject_relation(link_b, link_c, subj["id"], "supports", "TestActor", tmp_path)

    with _patch_cfg(tmp_path), _patch_root(tmp_path):
        with mock.patch("server._subjects.read_subject_memory", return_value="Content."):
            with mock.patch("server._subjects.parse_memory_view", return_value="Content."):
                result = server.recall_subject(id=subj["id"])

    assert "2 edges" in result
    assert "--updates-->" in result
    assert "--supports-->" in result


# ---------------------------------------------------------------------------
# Graceful degradation when db query fails
# ---------------------------------------------------------------------------

def test_recall_subject_relation_query_failure_does_not_crash(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _seed_subject(tmp_path, "Fail Subject")

    with _patch_cfg(tmp_path), _patch_root(tmp_path):
        with mock.patch("server._subjects.read_subject_memory", return_value="OK."):
            with mock.patch("server._subjects.parse_memory_view", return_value="OK."):
                with mock.patch(
                    "server._db.get_subject_relations_for_subject",
                    side_effect=RuntimeError("db exploded"),
                ):
                    result = server.recall_subject(id=subj["id"])

    # Should still return the memory content; history section just absent
    assert "OK." in result
    assert "Belief history" not in result
