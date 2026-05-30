"""Tests for the escalate_message memory-action handler."""

import json
from pathlib import Path
from unittest import mock

import db as _db
import daemon_escalate
import meta_agent
import subjects as _subjects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = {
    "escalation_token_threshold": 1,    # always pass threshold check in tests
    "escalation_token_max": 999999,
    "relevance_floor": 0.0,
}

_THREE_MESSAGES = [
    {"role": "user", "content": "Tell me about GPU rendering."},
    {"role": "assistant", "content": "GPU rendering uses the graphics card."},
    {"role": "user", "content": "Which GPUs are best for Blender?"},
]


def _seed_messages(tmp_path: Path, session_id: str, msgs: list[dict] | None = None) -> None:
    _db.init_db(tmp_path)
    _db.upsert_session_messages(
        session_id,
        msgs or _THREE_MESSAGES,
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )


def _msg_ids(tmp_path: Path, session_id: str, indices: list[int]) -> list[str]:
    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            "SELECT id, message_index FROM autobiographical_memory WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    id_by_index = {int(r["message_index"]): r["id"] for r in rows}
    return [id_by_index[i] for i in indices]


def _make_payload(session_id: str, message_ids: list[str]) -> dict:
    return {"session_id": session_id, "message_ids": message_ids}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_handler_assigns_messages_to_existing_subject(tmp_path: Path) -> None:
    """Handler fetches messages, calls escalate_unassigned, saves assignments, writes memory."""
    _seed_messages(tmp_path, "sess-e")
    subject = _subjects.create_subject("GPU Rendering", "GPU topics", None, tmp_path)
    ids = _msg_ids(tmp_path, "sess-e", [0, 1, 2])

    escalation_result = {
        "assignments": {0: subject["id"], 1: subject["id"], 2: None},
        "new_subjects": [],
    }

    with (
        mock.patch.object(meta_agent, "escalate_unassigned", return_value=escalation_result),
        mock.patch.object(meta_agent, "update_subject_memory", return_value=("view", "log", "active")),
        mock.patch.object(_subjects, "read_subject_memory", return_value=""),
        mock.patch.object(_subjects, "parse_memory_view", return_value=""),
        mock.patch.object(_subjects, "parse_memory_log", return_value=[]),
        mock.patch.object(_subjects, "compose_memory_file", return_value="composed"),
        mock.patch.object(_subjects, "write_subject_memory") as mock_write,
    ):
        daemon_escalate._handle_escalate_message(
            tmp_path, _CFG, None, _make_payload("sess-e", ids)
        )

    # Memory was written for the subject.
    mock_write.assert_called_once()

    # Assignments were saved for the two assigned messages.
    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            "SELECT am.message_index FROM semantic_links sl"
            " JOIN autobiographical_memory am ON sl.message_id = am.id"
            " WHERE am.session_id = 'sess-e'",
        ).fetchall()
    assigned_indices = {int(row["message_index"]) for row in rows}
    assert 0 in assigned_indices
    assert 1 in assigned_indices


def test_handler_creates_new_subject_and_assigns_messages(tmp_path: Path) -> None:
    """Handler creates a new subject from model output and assigns messages to it."""
    _seed_messages(tmp_path, "sess-new")
    ids = _msg_ids(tmp_path, "sess-new", [0, 1])

    escalation_result = {
        "assignments": {},
        "new_subjects": [
            {
                "name": "Render Engines",
                "description": "Discussion of render engines",
                "parent_id": None,
                "message_indices": [0, 1],
            }
        ],
    }

    with (
        mock.patch.object(meta_agent, "escalate_unassigned", return_value=escalation_result),
        mock.patch.object(meta_agent, "update_subject_memory", return_value=("view", "log", "active")),
        mock.patch.object(_subjects, "read_subject_memory", return_value=""),
        mock.patch.object(_subjects, "parse_memory_view", return_value=""),
        mock.patch.object(_subjects, "parse_memory_log", return_value=[]),
        mock.patch.object(_subjects, "compose_memory_file", return_value="composed"),
        mock.patch.object(_subjects, "write_subject_memory") as mock_write,
    ):
        daemon_escalate._handle_escalate_message(
            tmp_path, _CFG, None, _make_payload("sess-new", ids)
        )

    mock_write.assert_called_once()
    known = _subjects.load_subjects(tmp_path)
    names = {s["name"] for s in known}
    assert "Render Engines" in names


# ---------------------------------------------------------------------------
# Empty / below-threshold cases
# ---------------------------------------------------------------------------

def test_handler_empty_message_ids_is_noop(tmp_path: Path) -> None:
    """Empty message_ids list returns immediately without calling the heavy model."""
    _db.init_db(tmp_path)
    with mock.patch.object(meta_agent, "escalate_unassigned") as mock_esc:
        daemon_escalate._handle_escalate_message(
            tmp_path, _CFG, None, {"session_id": "sess-empty", "message_ids": []}
        )
    mock_esc.assert_not_called()


def test_handler_below_token_threshold_skips_heavy_model(tmp_path: Path) -> None:
    """When total tokens are below the threshold the heavy model is not called."""
    _seed_messages(tmp_path, "sess-small")
    ids = _msg_ids(tmp_path, "sess-small", [0])
    cfg = dict(_CFG)
    cfg["escalation_token_threshold"] = 999999  # make it unreachable

    with mock.patch.object(meta_agent, "escalate_unassigned") as mock_esc:
        daemon_escalate._handle_escalate_message(
            tmp_path, cfg, None, _make_payload("sess-small", ids)
        )
    mock_esc.assert_not_called()


# ---------------------------------------------------------------------------
# Hallucination guards
# ---------------------------------------------------------------------------

def test_handler_hallucinated_subject_id_is_skipped(tmp_path: Path) -> None:
    """A subject ID returned by the model that does not exist is silently skipped."""
    _seed_messages(tmp_path, "sess-hall")
    ids = _msg_ids(tmp_path, "sess-hall", [0])

    escalation_result = {
        "assignments": {0: "nonexistent-uuid"},
        "new_subjects": [],
    }

    with (
        mock.patch.object(meta_agent, "escalate_unassigned", return_value=escalation_result),
    ):
        # Should not raise.
        daemon_escalate._handle_escalate_message(
            tmp_path, _CFG, None, _make_payload("sess-hall", ids)
        )

    # No assignment should have been saved (hallucinated ID was skipped).
    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            "SELECT sl.message_id FROM semantic_links sl"
            " JOIN autobiographical_memory am ON sl.message_id = am.id"
            " WHERE am.session_id = 'sess-hall'",
        ).fetchall()
    assert len(rows) == 0


def test_handler_hallucinated_parent_id_creates_subject_at_root(tmp_path: Path) -> None:
    """A new subject with a hallucinated parent_id is created at root level instead."""
    _seed_messages(tmp_path, "sess-parent")
    ids = _msg_ids(tmp_path, "sess-parent", [0])

    escalation_result = {
        "assignments": {},
        "new_subjects": [
            {
                "name": "Orphaned Subject",
                "description": "Parent doesn't exist",
                "parent_id": "bogus-parent-id",
                "message_indices": [0],
            }
        ],
    }

    with (
        mock.patch.object(meta_agent, "escalate_unassigned", return_value=escalation_result),
        mock.patch.object(meta_agent, "update_subject_memory", return_value=("view", "log", "active")),
        mock.patch.object(_subjects, "read_subject_memory", return_value=""),
        mock.patch.object(_subjects, "parse_memory_view", return_value=""),
        mock.patch.object(_subjects, "parse_memory_log", return_value=[]),
        mock.patch.object(_subjects, "compose_memory_file", return_value="composed"),
        mock.patch.object(_subjects, "write_subject_memory"),
    ):
        daemon_escalate._handle_escalate_message(
            tmp_path, _CFG, None, _make_payload("sess-parent", ids)
        )

    known = _subjects.load_subjects(tmp_path)
    orphan = next((s for s in known if s["name"] == "Orphaned Subject"), None)
    assert orphan is not None, "Subject should have been created despite bad parent_id"
    assert orphan.get("parent_id") is None


# ---------------------------------------------------------------------------
# EscalateFailed re-raise
# ---------------------------------------------------------------------------

def test_handler_escalate_failed_reraises(tmp_path: Path) -> None:
    """EscalateFailed propagates so the queue marks the action for retry."""
    _seed_messages(tmp_path, "sess-fail")
    ids = _msg_ids(tmp_path, "sess-fail", [0, 1])

    with (
        mock.patch.object(
            meta_agent,
            "escalate_unassigned",
            side_effect=meta_agent.EscalateFailed("model error"),
        ),
    ):
        try:
            daemon_escalate._handle_escalate_message(
                tmp_path, _CFG, None, _make_payload("sess-fail", ids)
            )
            assert False, "Expected EscalateFailed to be raised"
        except meta_agent.EscalateFailed:
            pass  # expected
