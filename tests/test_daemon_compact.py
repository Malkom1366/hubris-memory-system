"""Tests for the compact_memory feature.

Covers:
  - has_pending_memory_action NULL dedup fix (db.py)
  - get_subjects_by_lru (db.py)
  - get_messages_fully_covered (db.py)
  - _count_session_tokens (daemon_watcher.py)
  - ContinueAdapter.write_tombstones (adapters.py)
  - _handle_compact_memory algorithm (daemon_compact.py)
  - compact_memory enqueue guard in _on_session_changed (daemon_watcher.py)
"""

import json
from pathlib import Path
from unittest import mock

import db as _db
import daemon_compact
import daemon_watcher
import subjects as _subjects
from frontend_adapters import ContinueAdapter, VSCodeCopilotAdapter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CFG = {
    "escalation_token_threshold": 999999,
    "escalation_token_max": 999999,
    "escalation_batch_interval": 1,
    "relevance_floor": 0.0,
    "context_compact_threshold": 80_000,
}


def _seed(tmp_path: Path, session_id: str, messages: list[dict]) -> None:
    _db.init_db(tmp_path)
    _db.upsert_session_messages(
        session_id, messages, "global", _db.ACTOR_MEMORY_WRITER, tmp_path
    )


def _link(
    tmp_path: Path,
    session_id: str,
    assignments: dict[int, dict[str, float]],
) -> None:
    """Save semantic links for the given session."""
    _db.save_assignments(
        session_id, assignments, _db.ACTOR_MESSAGE_CLASSIFIER, tmp_path
    )


# ---------------------------------------------------------------------------
# 1. has_pending_memory_action NULL dedup (bug fix)
# ---------------------------------------------------------------------------

def test_has_pending_null_subject_dedup_positive(tmp_path: Path) -> None:
    """After the IS fix, a pending compact_memory (NULL subject) is found."""
    _db.init_db(tmp_path)
    _db.enqueue_memory_action(
        action_type="compact_memory",
        subject_id=None,
        payload={},
        actor=_db.ACTOR_MEMORY_ACTIONS,
        root=tmp_path,
    )
    assert _db.has_pending_memory_action("compact_memory", None, tmp_path) is True


def test_has_pending_null_subject_dedup_negative(tmp_path: Path) -> None:
    """No pending compact_memory -> returns False for NULL subject."""
    _db.init_db(tmp_path)
    assert _db.has_pending_memory_action("compact_memory", None, tmp_path) is False


def test_has_pending_non_null_subject_unaffected(tmp_path: Path) -> None:
    """The IS fix does not break the non-NULL path."""
    _db.init_db(tmp_path)
    s = _subjects.create_subject("Test Subject", "desc", None, tmp_path)
    _db.enqueue_memory_action(
        action_type="finalize_subject",
        subject_id=s["id"],
        payload={},
        actor=_db.ACTOR_MEMORY_ACTIONS,
        root=tmp_path,
    )
    assert _db.has_pending_memory_action("finalize_subject", s["id"], tmp_path) is True
    assert _db.has_pending_memory_action("finalize_subject", "other-id", tmp_path) is False


# ---------------------------------------------------------------------------
# 2. get_subjects_by_lru
# ---------------------------------------------------------------------------

def test_get_subjects_by_lru_order(tmp_path: Path) -> None:
    """Subjects are returned coldest-first (lowest MAX message_index first)."""
    session_id = "sess-lru"
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
    _seed(tmp_path, session_id, msgs)

    cold_subj = _subjects.create_subject("Cold Subject", "desc", None, tmp_path)
    warm_subj = _subjects.create_subject("Warm Subject", "desc", None, tmp_path)

    # cold_subj last touched at index 1, warm_subj last touched at index 4
    _link(tmp_path, session_id, {
        0: {cold_subj["id"]: 1.0},
        1: {cold_subj["id"]: 1.0},
        3: {warm_subj["id"]: 1.0},
        4: {warm_subj["id"]: 1.0},
    })

    result = _db.get_subjects_by_lru(session_id, tmp_path)
    ids = [r["subject_id"] for r in result]
    assert ids == [cold_subj["id"], warm_subj["id"]]


def test_get_subjects_by_lru_empty(tmp_path: Path) -> None:
    """Returns empty list when no semantic links exist."""
    _db.init_db(tmp_path)
    result = _db.get_subjects_by_lru("no-such-session", tmp_path)
    assert result == []


def test_get_subjects_by_lru_last_msg_idx_correct(tmp_path: Path) -> None:
    """Each row includes the correct last_msg_idx."""
    session_id = "sess-idx"
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
    _seed(tmp_path, session_id, msgs)
    s = _subjects.create_subject("Solo Subject", "desc", None, tmp_path)
    _link(tmp_path, session_id, {2: {s["id"]: 1.0}, 4: {s["id"]: 1.0}})

    result = _db.get_subjects_by_lru(session_id, tmp_path)
    assert len(result) == 1
    assert result[0]["last_msg_idx"] == 4


# ---------------------------------------------------------------------------
# 3. get_messages_fully_covered
# ---------------------------------------------------------------------------

def test_get_messages_fully_covered_basic(tmp_path: Path) -> None:
    """A message whose only subject is in closing_ids is returned."""
    session_id = "sess-cov"
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(3)]
    _seed(tmp_path, session_id, msgs)
    s = _subjects.create_subject("SubjectA", "desc", None, tmp_path)
    _link(tmp_path, session_id, {0: {s["id"]: 1.0}, 1: {s["id"]: 1.0}})

    covered = _db.get_messages_fully_covered(session_id, {s["id"]}, tmp_path)
    assert covered == {0, 1}


def test_get_messages_fully_covered_excludes_shared(tmp_path: Path) -> None:
    """A message linked to a subject outside closing_ids is excluded."""
    session_id = "sess-share"
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(3)]
    _seed(tmp_path, session_id, msgs)
    s_in = _subjects.create_subject("In Closing", "desc", None, tmp_path)
    s_out = _subjects.create_subject("Out Closing", "desc", None, tmp_path)
    # msg 0: only s_in -> covered
    # msg 1: both -> excluded (s_out is outside)
    _link(tmp_path, session_id, {
        0: {s_in["id"]: 1.0},
        1: {s_in["id"]: 1.0, s_out["id"]: 1.0},
    })

    covered = _db.get_messages_fully_covered(session_id, {s_in["id"]}, tmp_path)
    assert covered == {0}


def test_get_messages_fully_covered_empty_closing(tmp_path: Path) -> None:
    """Empty closing set returns empty set."""
    session_id = "sess-empty-c"
    msgs = [{"role": "user", "content": "hi"}]
    _seed(tmp_path, session_id, msgs)
    s = _subjects.create_subject("S", "d", None, tmp_path)
    _link(tmp_path, session_id, {0: {s["id"]: 1.0}})

    covered = _db.get_messages_fully_covered(session_id, set(), tmp_path)
    assert covered == set()


def test_get_messages_fully_covered_unlinked_messages_excluded(tmp_path: Path) -> None:
    """Messages with no semantic links at all are never returned."""
    session_id = "sess-unlinked"
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(3)]
    _seed(tmp_path, session_id, msgs)
    s = _subjects.create_subject("S", "d", None, tmp_path)
    # Only link msg 0; msgs 1 and 2 are unlinked (null-pool)
    _link(tmp_path, session_id, {0: {s["id"]: 1.0}})

    covered = _db.get_messages_fully_covered(session_id, {s["id"]}, tmp_path)
    assert covered == {0}


# ---------------------------------------------------------------------------
# 4. _count_session_tokens
# ---------------------------------------------------------------------------

def test_count_session_tokens_empty(tmp_path: Path) -> None:
    assert daemon_watcher._count_session_tokens([]) == 0


def test_count_session_tokens_simple(tmp_path: Path) -> None:
    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hello there"},
    ]
    count = daemon_watcher._count_session_tokens(messages)
    assert count > 0
    assert count < 50  # sanity: well under for two short messages


def test_count_session_tokens_accumulates(tmp_path: Path) -> None:
    short = [{"role": "user", "content": "hi"}]
    long_msgs = [{"role": "user", "content": "hi " * 100}]
    assert daemon_watcher._count_session_tokens(long_msgs) > daemon_watcher._count_session_tokens(short)


# ---------------------------------------------------------------------------
# 5. ContinueAdapter.write_tombstones
# ---------------------------------------------------------------------------

def _make_session_file(tmp_path: Path, session_id: str, messages: list[dict]) -> Path:
    """Write a Continue-format session JSON file and return its path."""
    turns = [
        {"message": {"role": m["role"], "content": m["content"]}, "contextItems": []}
        for m in messages
    ]
    path = tmp_path / f"{session_id}.json"
    path.write_text(json.dumps({"history": turns}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def test_write_tombstones_replaces_content(tmp_path: Path) -> None:
    """write_tombstones replaces the content of the specified messages in-place."""
    session_id = "sess-ts"
    messages = [
        {"role": "user", "content": "Message zero"},
        {"role": "user", "content": "Message one"},
        {"role": "user", "content": "Message two"},
    ]
    _make_session_file(tmp_path, session_id, messages)
    adapter = ContinueAdapter(tmp_path)

    adapter.write_tombstones(session_id, {1: "[HUBRIS-COMPACT] Tombstone."})

    result = adapter.read_messages(session_id)
    assert result[0]["content"] == "Message zero"
    assert result[1]["content"] == "[HUBRIS-COMPACT] Tombstone."
    assert result[2]["content"] == "Message two"


def test_write_tombstones_preserves_turn_count(tmp_path: Path) -> None:
    """Tombstoning does not add or remove turns."""
    session_id = "sess-ts-count"
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
    _make_session_file(tmp_path, session_id, messages)
    adapter = ContinueAdapter(tmp_path)

    adapter.write_tombstones(session_id, {0: "[COMPACT]", 3: "[COMPACT]"})

    result = adapter.read_messages(session_id)
    assert len(result) == 5


def test_write_tombstones_skips_anchor_in_index_mapping(tmp_path: Path) -> None:
    """The anchor turn at position 0 is not counted as a HuBrIS message index."""
    session_id = "sess-ts-anchor"
    anchor_content = "[HUBRIS-CATALOG-ANCHOR]\nsome catalog"
    messages_with_anchor = [
        {"role": "user", "content": anchor_content},
        {"role": "user", "content": "Real message 0"},   # HuBrIS index 0
        {"role": "assistant", "content": "Real message 1"},  # HuBrIS index 1
    ]
    _make_session_file(tmp_path, session_id, messages_with_anchor)
    adapter = ContinueAdapter(tmp_path)

    # Tombstone HuBrIS index 0 (turn 1 in the file, after the anchor)
    adapter.write_tombstones(session_id, {0: "[COMPACT] Real message 0 tombstoned."})

    result = adapter.read_messages(session_id)
    # Anchor is stripped by read_messages - only 2 real messages remain.
    assert len(result) == 2
    # Turn 1 (HuBrIS index 0) was tombstoned.
    assert result[0]["content"] == "[COMPACT] Real message 0 tombstoned."
    # Turn 2 (HuBrIS index 1) is untouched.
    assert result[1]["content"] == "Real message 1"


def test_write_tombstones_noop_on_missing_session(tmp_path: Path) -> None:
    """write_tombstones silently does nothing if the session file does not exist."""
    adapter = ContinueAdapter(tmp_path)
    # Should not raise
    adapter.write_tombstones("nonexistent-session", {0: "tombstone"})


def test_write_tombstones_noop_for_vscode_adapter(tmp_path: Path) -> None:
    """VSCodeCopilotAdapter.write_tombstones is a no-op (read-only adapter)."""
    adapter = VSCodeCopilotAdapter(tmp_path)
    # Should not raise and should not write anything
    adapter.write_tombstones("any-session", {0: "tombstone"})


# ---------------------------------------------------------------------------
# 6. _handle_compact_memory algorithm
# ---------------------------------------------------------------------------

def _setup_compact_scenario(tmp_path: Path) -> tuple[str, list[dict], dict, dict]:
    """
    Build a session with 10 messages, two subjects, and return
    (session_id, messages, cold_subject, warm_subject).

    cold_subject: last touched at message 2
    warm_subject: last touched at message 8
    """
    session_id = "sess-compact"
    messages = [{"role": "user", "content": f"message content {i} " + ("x" * 20)} for i in range(10)]
    _seed(tmp_path, session_id, messages)

    cold = _subjects.create_subject("Cold Subject", "GPU basics", None, tmp_path)
    warm = _subjects.create_subject("Warm Subject", "GPU advanced", None, tmp_path)

    _link(tmp_path, session_id, {
        0: {cold["id"]: 1.0},
        1: {cold["id"]: 1.0},
        2: {cold["id"]: 1.0},
        6: {warm["id"]: 1.0},
        7: {warm["id"]: 1.0},
        8: {warm["id"]: 1.0},
    })
    return session_id, messages, cold, warm


def test_handler_archives_cold_subjects(tmp_path: Path) -> None:
    """Handler archives subjects in closing set."""
    session_id, messages, cold, warm = _setup_compact_scenario(tmp_path)

    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": m["role"], "content": m["content"]}, "contextItems": []} for m in messages]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    adapter = ContinueAdapter(tmp_path)

    with mock.patch("subjects.set_subject_state") as mock_archive:
        daemon_compact._handle_compact_memory(
            tmp_path, _CFG, None,
            {"session_id": session_id, "threshold": 5}  # low threshold: stop after cold
        )
        # cold subject should have been archived
        archived_ids = {call.args[0] for call in mock_archive.call_args_list}
        assert cold["id"] in archived_ids


def test_handler_writes_tombstones_for_covered_messages(tmp_path: Path) -> None:
    """Handler writes tombstones for messages whose subjects are all in closing set."""
    session_id, messages, cold, warm = _setup_compact_scenario(tmp_path)

    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": m["role"], "content": m["content"]}, "contextItems": []} for m in messages]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    adapter = ContinueAdapter(tmp_path)

    with mock.patch("daemon_compact._adapter", ContinueAdapter(tmp_path)):
        daemon_compact._handle_compact_memory(
            tmp_path, _CFG, None,
            {"session_id": session_id, "threshold": 5}
        )

    # Read back the session file; messages 0-2 should be tombstoned
    result = adapter.read_messages(session_id)
    for idx in [0, 1, 2]:
        assert "[HUBRIS-COMPACT]" in result[idx]["content"], (
            f"Expected tombstone at index {idx}, got: {result[idx]['content']!r}"
        )
    # Messages 6-8 (warm subject) should be untouched
    for idx in [6, 7, 8]:
        assert "[HUBRIS-COMPACT]" not in result[idx]["content"]


def test_handler_zero_yield_subject_still_added_to_closing(tmp_path: Path) -> None:
    """
    A subject whose messages are all shared with warmer subjects adds
    0 new tombstonable messages but stays in the closing set so that
    co-subjects can unlock the shared messages later.
    """
    session_id = "sess-zero-yield"
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(4)]
    _seed(tmp_path, session_id, messages)

    cold = _subjects.create_subject("Cold Zero Yield", "desc", None, tmp_path)
    warm = _subjects.create_subject("Warm Co-Subject", "desc", None, tmp_path)

    # msg 0 and 1 are shared between cold AND warm
    # cold has no exclusive messages
    _link(tmp_path, session_id, {
        0: {cold["id"]: 1.0, warm["id"]: 1.0},
        1: {cold["id"]: 1.0, warm["id"]: 1.0},
        2: {warm["id"]: 1.0},
    })

    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": m["role"], "content": m["content"]}, "contextItems": []} for m in messages]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    with mock.patch("daemon_compact._adapter", ContinueAdapter(tmp_path)):
        daemon_compact._handle_compact_memory(
            tmp_path, _CFG, None,
            {"session_id": session_id, "threshold": 5}
        )

    adapter = ContinueAdapter(tmp_path)
    result = adapter.read_messages(session_id)
    # Once both cold and warm are in closing, msgs 0 and 1 unlock
    for idx in [0, 1]:
        assert "[HUBRIS-COMPACT]" in result[idx]["content"], (
            f"Shared message {idx} should be tombstoned once both subjects in closing"
        )


def test_handler_stops_at_threshold(tmp_path: Path) -> None:
    """Handler stops iterating after enough subjects to clear the threshold."""
    session_id, messages, cold, warm = _setup_compact_scenario(tmp_path)

    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": m["role"], "content": m["content"]}, "contextItems": []} for m in messages]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    # Use a threshold of 0 (always satisfied after first real yield)
    with mock.patch("daemon_compact._adapter", ContinueAdapter(tmp_path)):
        daemon_compact._handle_compact_memory(
            tmp_path, _CFG, None,
            {"session_id": session_id, "threshold": 0}
        )

    adapter = ContinueAdapter(tmp_path)
    result = adapter.read_messages(session_id)

    # Cold messages (0-2) should be tombstoned
    for idx in [0, 1, 2]:
        assert "[HUBRIS-COMPACT]" in result[idx]["content"]
    # Warm messages (6-8) should NOT be tombstoned - loop stopped at threshold
    for idx in [6, 7, 8]:
        assert "[HUBRIS-COMPACT]" not in result[idx]["content"]


def test_handler_exhausts_all_subjects_if_threshold_never_met(tmp_path: Path) -> None:
    """If threshold is never met, all subjects are processed (best-effort)."""
    session_id, messages, cold, warm = _setup_compact_scenario(tmp_path)

    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": m["role"], "content": m["content"]}, "contextItems": []} for m in messages]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    # Threshold is 99999 tokens - will never be met with these tiny messages
    with mock.patch("daemon_compact._adapter", ContinueAdapter(tmp_path)):
        daemon_compact._handle_compact_memory(
            tmp_path, _CFG, None,
            {"session_id": session_id, "threshold": 99999}
        )

    adapter = ContinueAdapter(tmp_path)
    result = adapter.read_messages(session_id)

    # All linked messages should be tombstoned (best-effort - processed all subjects)
    for idx in [0, 1, 2, 6, 7, 8]:
        assert "[HUBRIS-COMPACT]" in result[idx]["content"]


# ---------------------------------------------------------------------------
# 7. compact_memory enqueue guard in _on_session_changed
# ---------------------------------------------------------------------------

def test_compact_memory_enqueued_when_threshold_exceeded(tmp_path: Path) -> None:
    """
    When _on_session_changed detects token count above threshold, it enqueues
    a compact_memory action.
    """
    session_id = "sess-enqueue"
    # Create a session file for a ContinueAdapter
    messages = [{"role": "user", "content": "hello world"}]
    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": "user", "content": "hello world"}, "contextItems": []}]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    adapter = ContinueAdapter(tmp_path)

    cfg = dict(_CFG)
    cfg["context_compact_threshold"] = 0  # always triggers
    cfg["workspace_id"] = "global"

    # Patch everything that touches external state
    with (
        mock.patch("daemon_watcher._config") as mock_cfg,
        mock.patch("daemon_watcher._db") as mock_db_mod,
        mock.patch("frontend_adapter_continue.catalog") as mock_catalog,
        mock.patch("daemon_watcher._subjects") as mock_subj,
        mock.patch("daemon_watcher._save_counts"),
        mock.patch("daemon_watcher._save_session_whitelist"),
        mock.patch("daemon_watcher._session_whitelist", new={session_id}),
        mock.patch("daemon_watcher._workspace_blacklist", new=set()),
        mock.patch("daemon_watcher._bound_session", new=None),
        mock.patch("daemon_watcher._state_lock"),
        mock.patch("daemon_watcher._session_message_counts", new={}),
        mock.patch("daemon_watcher._count_session_tokens", return_value=100_000),
    ):
        mock_cfg.load.return_value = cfg
        mock_cfg.memory_root.return_value = tmp_path
        mock_db_mod.upsert_session_messages.return_value = None
        mock_db_mod.has_pending_memory_action.return_value = False
        mock_db_mod.enqueue_memory_action.return_value = 1
        mock_db_mod.ACTOR_MEMORY_WRITER = _db.ACTOR_MEMORY_WRITER
        mock_catalog.load_catalog.return_value = {}
        mock_catalog.ANCHOR_MARKER = "[HUBRIS-CATALOG-ANCHOR]"
        mock_catalog.check_anchor.return_value = True
        mock_catalog.rebuild_catalog_from_subjects.return_value = {}
        mock_subj.load_subjects.return_value = []

        daemon_watcher._on_session_changed(session_id, adapter)

        # compact_memory should have been enqueued
        enqueue_calls = [
            call for call in mock_db_mod.enqueue_memory_action.call_args_list
            if (len(call.args) > 0 and call.args[0] == "compact_memory")
            or call.kwargs.get("action_type") == "compact_memory"
        ]
        assert len(enqueue_calls) >= 1, "Expected compact_memory to be enqueued"


def test_compact_memory_not_enqueued_when_already_pending(tmp_path: Path) -> None:
    """
    When a compact_memory action is already pending, _on_session_changed
    does not enqueue a second one.
    """
    session_id = "sess-dedup"
    session_file_path = tmp_path / f"{session_id}.json"
    turns = [{"message": {"role": "user", "content": "hello"}, "contextItems": []}]
    session_file_path.write_text(json.dumps({"history": turns}, indent=2), encoding="utf-8")

    adapter = ContinueAdapter(tmp_path)

    cfg = dict(_CFG)
    cfg["context_compact_threshold"] = 0
    cfg["workspace_id"] = "global"

    with (
        mock.patch("daemon_watcher._config") as mock_cfg,
        mock.patch("daemon_watcher._db") as mock_db_mod,
        mock.patch("frontend_adapter_continue.catalog") as mock_catalog,
        mock.patch("daemon_watcher._subjects") as mock_subj,
        mock.patch("daemon_watcher._save_counts"),
        mock.patch("daemon_watcher._save_session_whitelist"),
        mock.patch("daemon_watcher._session_whitelist", new={session_id}),
        mock.patch("daemon_watcher._workspace_blacklist", new=set()),
        mock.patch("daemon_watcher._bound_session", new=None),
        mock.patch("daemon_watcher._state_lock"),
        mock.patch("daemon_watcher._session_message_counts", new={}),
        mock.patch("daemon_watcher._count_session_tokens", return_value=100_000),
    ):
        mock_cfg.load.return_value = cfg
        mock_cfg.memory_root.return_value = tmp_path
        mock_db_mod.upsert_session_messages.return_value = None
        mock_db_mod.has_pending_memory_action.return_value = True  # already pending
        mock_db_mod.enqueue_memory_action.return_value = 1
        mock_db_mod.ACTOR_MEMORY_WRITER = _db.ACTOR_MEMORY_WRITER
        mock_catalog.load_catalog.return_value = {}
        mock_catalog.ANCHOR_MARKER = "[HUBRIS-CATALOG-ANCHOR]"
        mock_catalog.check_anchor.return_value = True
        mock_catalog.rebuild_catalog_from_subjects.return_value = {}
        mock_subj.load_subjects.return_value = []

        daemon_watcher._on_session_changed(session_id, adapter)

        enqueue_calls = [
            call for call in mock_db_mod.enqueue_memory_action.call_args_list
            if (len(call.args) > 0 and call.args[0] == "compact_memory")
            or call.kwargs.get("action_type") == "compact_memory"
        ]
        assert len(enqueue_calls) == 0, "compact_memory should NOT be enqueued when already pending"
