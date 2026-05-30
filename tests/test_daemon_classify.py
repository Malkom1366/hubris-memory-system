"""Tests for the classify_message memory-action handler."""

from pathlib import Path
from unittest import mock

import db as _db
import daemon_classify
import meta_agent
import subjects as _subjects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = {
    "classify_failure_threshold": 3,
    "escalation_token_threshold": 999999,  # effectively disable escalation
    "escalation_token_max": 999999,
    "escalation_batch_interval": 1,
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


def _make_payload(session_id: str, from_index: int, to_index: int) -> dict:
    return {
        "session_id": session_id,
        "from_index": from_index,
        "to_index": to_index,
        "workspace_id": "global",
    }


# ---------------------------------------------------------------------------
# classify_message handler: subjects present
# ---------------------------------------------------------------------------

def test_handler_assigns_messages_and_advances_hwm(tmp_path: Path) -> None:
    """Handler writes semantic links and advances committed HWM when subjects exist."""
    _seed_messages(tmp_path, "sess-a")
    subject = _subjects.create_subject("GPU Rendering", "GPU topics", None, tmp_path)

    batch_result: dict[int, dict[str, float]] = {
        0: {subject["id"]: 0.9},
        1: {subject["id"]: 0.8},
        2: {subject["id"]: 0.85},
    }

    def fake_classify(messages, subjects, cfg, on_batch_done):
        on_batch_done(batch_result, len(messages))

    with (
        mock.patch.object(meta_agent, "classify_messages", side_effect=fake_classify),
        mock.patch.object(meta_agent, "update_subject_memory", return_value=("view", "log", "active")),
        mock.patch.object(_subjects, "read_subject_memory", return_value=""),
        mock.patch.object(_subjects, "parse_memory_view", return_value=""),
        mock.patch.object(_subjects, "parse_memory_log", return_value=[]),
        mock.patch.object(_subjects, "compose_memory_file", return_value="composed"),
        mock.patch.object(_subjects, "write_subject_memory"),
    ):
        daemon_classify._handle_classify_message(tmp_path, _CFG, None, _make_payload("sess-a", 0, 3))

    # Committed HWM must have advanced to to_index.
    with daemon_classify._state_lock:
        committed = daemon_classify._committed_counts.get("sess-a", -1)
    assert committed == 3

    # Semantic links must exist in the DB.
    with _db.connect(tmp_path) as conn:
        links = conn.execute(
            "SELECT message_id FROM semantic_links WHERE subject_id = ?",
            (subject["id"],),
        ).fetchall()
    assert len(links) == 3


# ---------------------------------------------------------------------------
# classify_message handler: no subjects - null pool path
# ---------------------------------------------------------------------------

def test_handler_no_subjects_adds_to_null_pool_and_advances_hwm(tmp_path: Path) -> None:
    """With no subjects all messages are null-assigned and HWM advances."""
    _seed_messages(tmp_path, "sess-b")

    with mock.patch.object(meta_agent, "classify_messages") as mock_classify:
        daemon_classify._handle_classify_message(tmp_path, _CFG, None, _make_payload("sess-b", 0, 3))
        # classify_messages must NOT be called when there are no subjects.
        mock_classify.assert_not_called()

    with daemon_classify._state_lock:
        committed = daemon_classify._committed_counts.get("sess-b", -1)
    assert committed == 3


# ---------------------------------------------------------------------------
# classify_message handler: ClassifyFailed below threshold - re-raises
# ---------------------------------------------------------------------------

def test_handler_classify_failed_below_threshold_reraises(tmp_path: Path) -> None:
    """ClassifyFailed below the auto-blacklist threshold must propagate so the queue retries."""
    _seed_messages(tmp_path, "sess-c")
    _subjects.create_subject("Any Subject", "desc", None, tmp_path)

    with daemon_classify._state_lock:
        daemon_classify._committed_counts["sess-c"] = 0
        daemon_classify._classify_failures.clear()

    with mock.patch.object(meta_agent, "classify_messages", side_effect=meta_agent.ClassifyFailed("boom")):
        try:
            daemon_classify._handle_classify_message(tmp_path, _CFG, None, _make_payload("sess-c", 0, 3))
        except meta_agent.ClassifyFailed:
            pass
        else:
            raise AssertionError("Expected ClassifyFailed to be re-raised")

    # Failure counter must have been incremented.
    with daemon_classify._state_lock:
        key = f"sess-c:0"
        assert daemon_classify._classify_failures.get(key, 0) == 1


# ---------------------------------------------------------------------------
# classify_message handler: ClassifyFailed at threshold - auto-blacklist
# ---------------------------------------------------------------------------

def test_handler_classify_failed_at_threshold_blacklists_and_returns(tmp_path: Path) -> None:
    """After reaching the threshold the handler blacklists the batch and returns cleanly."""
    _seed_messages(tmp_path, "sess-d")
    _subjects.create_subject("Any Subject", "desc", None, tmp_path)

    threshold = 3
    cfg = dict(_CFG, classify_failure_threshold=threshold)

    # Pre-seed failure counter so this call hits the threshold.
    with daemon_classify._state_lock:
        daemon_classify._committed_counts["sess-d"] = 0
        daemon_classify._classify_failures[f"sess-d:0"] = threshold - 1
        daemon_classify._message_blacklist.pop("sess-d", None)

    with mock.patch.object(meta_agent, "classify_messages", side_effect=meta_agent.ClassifyFailed("boom")):
        # Should NOT raise - action completes normally.
        daemon_classify._handle_classify_message(tmp_path, cfg, None, _make_payload("sess-d", 0, 3))

    with daemon_classify._state_lock:
        blacklist = daemon_classify._message_blacklist.get("sess-d", set())
        committed = daemon_classify._committed_counts.get("sess-d", 0)

    assert len(blacklist) > 0, "Expected blacklisted indices"
    assert committed > 0, "Expected HWM to advance past blacklisted batch"

    # Failure entry must be cleared after auto-blacklist.
    with daemon_classify._state_lock:
        assert f"sess-d:0" not in daemon_classify._classify_failures


# ---------------------------------------------------------------------------
# classify_message handler: empty delta is a no-op
# ---------------------------------------------------------------------------

def test_handler_empty_delta_is_noop(tmp_path: Path) -> None:
    """from_index == to_index means no new messages; handler returns without touching DB."""
    _seed_messages(tmp_path, "sess-e")

    with mock.patch.object(meta_agent, "classify_messages") as mock_classify:
        daemon_classify._handle_classify_message(tmp_path, _CFG, None, _make_payload("sess-e", 0, 0))
        mock_classify.assert_not_called()


# ---------------------------------------------------------------------------
# memory_classifier thread smoke test
# ---------------------------------------------------------------------------

def test_memory_classifier_thread_drains_enqueued_action(tmp_path: Path) -> None:
    """Classify handler processes a claimed action and marks it complete."""
    _seed_messages(tmp_path, "sess-thread")
    _subjects.create_subject("Thread Subject", "desc", None, tmp_path)

    action_id = _db.enqueue_memory_action(
        action_type="classify_message",
        subject_id=None,
        payload=_make_payload("sess-thread", 0, 3),
        actor=_db.ACTOR_MEMORY_WRITER,
        root=tmp_path,
    )

    def fake_classify(messages, subjects, cfg, on_batch_done):
        on_batch_done({}, len(messages))

    with (
        mock.patch.object(meta_agent, "classify_messages", side_effect=fake_classify),
        mock.patch.object(meta_agent, "update_subject_memory", return_value=("v", "l", "active")),
        mock.patch.object(_subjects, "read_subject_memory", return_value=""),
        mock.patch.object(_subjects, "parse_memory_view", return_value=""),
        mock.patch.object(_subjects, "parse_memory_log", return_value=[]),
        mock.patch.object(_subjects, "compose_memory_file", return_value="c"),
        mock.patch.object(_subjects, "write_subject_memory"),
    ):
        action = _db.claim_next_memory_action(root=tmp_path)
        assert action is not None
        daemon_classify._handle_classify_message(
            tmp_path, dict(_CFG), action["subject_id"], action["payload"]
        )
        _db.complete_memory_action(action["id"], root=tmp_path)

    # complete_memory_action DELETEs the row on success - absence means it completed.
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT status FROM memory_actions WHERE id = ?", (action_id,)
        ).fetchone()
    assert row is None, "Action row should be deleted after successful completion"


# ---------------------------------------------------------------------------
# Escalation enqueue
# ---------------------------------------------------------------------------

def test_handler_enqueues_escalate_message_for_null_assigned_messages(tmp_path: Path) -> None:
    """Handler enqueues an escalate_message action for any messages left null-assigned."""
    _seed_messages(tmp_path, "sess-esc")
    subject = _subjects.create_subject("GPU Rendering", "GPU topics", None, tmp_path)

    # Only message 0 gets assigned; messages 1 and 2 stay null.
    batch_result: dict[int, dict[str, float]] = {
        0: {subject["id"]: 0.9},
        1: {},
        2: {},
    }

    def fake_classify(messages, subjects, cfg, on_batch_done):
        on_batch_done(batch_result, len(messages))

    with (
        mock.patch.object(meta_agent, "classify_messages", side_effect=fake_classify),
        mock.patch.object(meta_agent, "update_subject_memory", return_value=("view", "log", "active")),
        mock.patch.object(_subjects, "read_subject_memory", return_value=""),
        mock.patch.object(_subjects, "parse_memory_view", return_value=""),
        mock.patch.object(_subjects, "parse_memory_log", return_value=[]),
        mock.patch.object(_subjects, "compose_memory_file", return_value="composed"),
        mock.patch.object(_subjects, "write_subject_memory"),
    ):
        daemon_classify._handle_classify_message(tmp_path, _CFG, None, _make_payload("sess-esc", 0, 3))

    # An escalate_message action must have been enqueued.
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM memory_actions WHERE action_type = 'escalate_message'",
        ).fetchone()
    assert row is not None, "Expected an escalate_message action to be enqueued"

    import json
    enqueued_payload = json.loads(row["payload_json"])
    assert enqueued_payload["session_id"] == "sess-esc"
    null_ids = enqueued_payload["message_ids"]
    # Messages at absolute indices 1 and 2 should be in the null list.
    with _db.connect(tmp_path) as conn:
        _rows = conn.execute(
            "SELECT id, message_index FROM autobiographical_memory WHERE session_id = 'sess-esc'",
        ).fetchall()
    _id_by_index = {int(r["message_index"]): r["id"] for r in _rows}
    assert _id_by_index[1] in null_ids
    assert _id_by_index[2] in null_ids
    assert _id_by_index[0] not in null_ids, "Assigned message should not be in null list"


def test_handler_no_subjects_enqueues_all_messages_for_escalation(tmp_path: Path) -> None:
    """With no subjects all messages are null-assigned and all IDs are handed to escalation."""
    _seed_messages(tmp_path, "sess-esc-all")

    daemon_classify._handle_classify_message(tmp_path, _CFG, None, _make_payload("sess-esc-all", 0, 3))

    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM memory_actions WHERE action_type = 'escalate_message'",
        ).fetchone()
    assert row is not None

    import json
    enqueued_payload = json.loads(row["payload_json"])
    with _db.connect(tmp_path) as conn:
        _rows = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = 'sess-esc-all'",
        ).fetchall()
    expected_ids = {r["id"] for r in _rows}
    assert set(enqueued_payload["message_ids"]) == expected_ids


# ---------------------------------------------------------------------------
# _scan_unclassified_sessions scanner
# ---------------------------------------------------------------------------


def test_scan_returns_item_for_session_with_unclassified_messages(tmp_path: Path) -> None:
    """Scanner yields an item when AB records exceed the committed HWM."""
    _seed_messages(tmp_path, "scan-sess-a")  # 3 messages, HWM defaults to 0

    items = daemon_classify._scan_unclassified_sessions(tmp_path)

    assert len(items) == 1
    item = items[0]
    assert item["subject_id"] == "scan-sess-a"
    assert item["payload"]["session_id"] == "scan-sess-a"
    assert item["payload"]["from_index"] == 0
    assert item["payload"]["to_index"] == 3


def test_scan_returns_nothing_when_session_fully_classified(tmp_path: Path) -> None:
    """Scanner yields nothing when committed HWM equals total message count."""
    _seed_messages(tmp_path, "scan-sess-b")  # 3 messages
    _db.save_committed_counts({"scan-sess-b": 3}, _db.ACTOR_SESSION_TRACKER, tmp_path)

    items = daemon_classify._scan_unclassified_sessions(tmp_path)

    assert items == []


def test_scan_returns_nothing_for_empty_db(tmp_path: Path) -> None:
    """Scanner yields nothing when there are no AB records at all."""
    _db.init_db(tmp_path)

    items = daemon_classify._scan_unclassified_sessions(tmp_path)

    assert items == []


def test_scan_reflects_partial_hwm(tmp_path: Path) -> None:
    """Scanner reports the correct from_index when a session is partially classified."""
    _seed_messages(tmp_path, "scan-sess-c")  # 3 messages
    _db.save_committed_counts({"scan-sess-c": 1}, _db.ACTOR_SESSION_TRACKER, tmp_path)

    items = daemon_classify._scan_unclassified_sessions(tmp_path)

    assert len(items) == 1
    assert items[0]["payload"]["from_index"] == 1
    assert items[0]["payload"]["to_index"] == 3

