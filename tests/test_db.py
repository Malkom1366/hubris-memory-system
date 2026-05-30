"""Tests for the SQLite-backed storage layer."""

from pathlib import Path

import db as _db
import embeddings as _emb
import subjects as _subjects


def test_init_db_creates_database_and_core_tables(tmp_path: Path) -> None:
    _db.init_db(tmp_path)

    assert _db.db_path(tmp_path).exists()

    with _db.connect(tmp_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "autobiographical_memory" in tables
    assert "semantic_subjects" in tables
    assert "semantic_links" in tables
    assert "audit_log" in tables


def test_audit_log_tracks_subject_changes_without_memory_payload(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Audit Topic", "desc", None, tmp_path)
    _subjects.write_subject_memory(subject, "Large memory payload should stay out of audit.", tmp_path)
    _subjects.set_subject_state(subject["id"], "dormant", tmp_path)

    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            """
            SELECT operation, new_values
            FROM audit_log
            WHERE table_name = 'semantic_subjects' AND record_id = ?
            ORDER BY id
            """,
            (subject["id"],),
        ).fetchall()

    assert [row["operation"] for row in rows] == ["INSERT", "UPDATE", "UPDATE"]
    assert all("memory_content" not in (row["new_values"] or "") for row in rows)


def test_one_message_can_link_to_multiple_subjects(tmp_path: Path) -> None:
    _db.upsert_session_messages(
        "session-1",
        [{"role": "user", "content": "Python async performance with Redis"}],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    subject_a = _subjects.create_subject("Python Async", "desc", None, tmp_path)
    subject_b = _subjects.create_subject("Redis Caching", "desc", None, tmp_path)

    # Two-subject link with distinct confidences via the public API.
    _db.save_assignments(
        "session-1",
        {0: {subject_a["id"]: 0.95, subject_b["id"]: 0.74}},
        _db.ACTOR_MESSAGE_CLASSIFIER,
        tmp_path,
    )

    with _db.connect(tmp_path) as conn:
        _id_row = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = 'session-1' AND message_index = 0",
        ).fetchone()
    message_id = _id_row["id"]
    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            "SELECT subject_id, confidence FROM semantic_links WHERE message_id = ? ORDER BY confidence DESC",
            (message_id,),
        ).fetchall()

    assert len(rows) == 2
    assert rows[0]["subject_id"] == subject_a["id"]
    assert abs(rows[0]["confidence"] - 0.95) < 1e-9
    assert rows[1]["subject_id"] == subject_b["id"]
    assert abs(rows[1]["confidence"] - 0.74) < 1e-9

    # load_assignments returns the primary (highest-confidence) subject only.
    primary = _db.load_assignments("session-1", tmp_path)
    assert primary == {0: subject_a["id"]}

    # load_all_assignments returns every link sorted by confidence desc.
    all_links = _db.load_all_assignments("session-1", tmp_path)
    assert all_links[0] == [(subject_a["id"], 0.95), (subject_b["id"], 0.74)]


def test_save_assignments_round_trips_multi_subject_relevance(tmp_path: Path) -> None:
    _db.upsert_session_messages(
        "session-2",
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    subject_a = _subjects.create_subject("Topic A", "desc", None, tmp_path)
    subject_b = _subjects.create_subject("Topic B", "desc", None, tmp_path)

    _db.save_assignments(
        "session-2",
        {
            0: {subject_a["id"]: 1.0},
            1: {subject_a["id"]: 0.6, subject_b["id"]: 0.9},
            2: None,
        },
        _db.ACTOR_MESSAGE_CLASSIFIER,
        tmp_path,
    )

    # Primary view: highest-confidence subject per message; index 2 unlinked.
    primary = _db.load_assignments("session-2", tmp_path)
    assert primary == {0: subject_a["id"], 1: subject_b["id"], 2: None}

    # Full view: index 1 has both subjects, ordered by confidence desc.
    all_links = _db.load_all_assignments("session-2", tmp_path)
    assert all_links[0] == [(subject_a["id"], 1.0)]
    assert all_links[1] == [(subject_b["id"], 0.9), (subject_a["id"], 0.6)]
    assert all_links.get(2, []) == []


def test_save_assignments_scoped_delete_preserves_untouched_indices(tmp_path: Path) -> None:
    """
    A sparse save_assignments call must only clear links for the indices it
    references. Links for messages outside the dict must survive.
    """
    _db.upsert_session_messages(
        "session-3",
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    subject = _subjects.create_subject("S", "", None, tmp_path)

    _db.save_assignments(
        "session-3",
        {0: {subject["id"]: 0.8}, 1: {subject["id"]: 0.4}},
        _db.ACTOR_MESSAGE_CLASSIFIER,
        tmp_path,
    )
    # Sparse update: only index 0 is rewritten. Index 1 must keep its link.
    _db.save_assignments(
        "session-3",
        {0: {subject["id"]: 1.0}},
        _db.ACTOR_MESSAGE_CLASSIFIER,
        tmp_path,
    )

    primary = _db.load_assignments("session-3", tmp_path)
    assert primary == {0: subject["id"], 1: subject["id"]}


def test_vec_messages_table_created_when_sqlite_vec_available(tmp_path: Path) -> None:
    """init_db() should create the vec_messages virtual table when sqlite-vec is installed."""
    if not _emb.is_available():
        import pytest  # type: ignore[import]
        pytest.skip("sqlite-vec not installed")

    _db.init_db(tmp_path)

    conn, loaded = _db.connect_vec(tmp_path)
    assert loaded, "sqlite-vec extension should load successfully"
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow', 'virtual')"
            ).fetchall()
        }
        assert "vec_messages" in tables
    finally:
        conn.close()


def test_upsert_and_search_embeddings_round_trip(tmp_path: Path) -> None:
    """Embeddings inserted via upsert_message_embeddings_batch can be found by semantic_search_messages."""
    import struct

    if not _emb.is_available():
        import pytest  # type: ignore[import]
        pytest.skip("sqlite-vec not installed")

    _db.init_db(tmp_path)
    _db.upsert_session_messages(
        "session-vec",
        [{"role": "user", "content": "GPU render job started"}],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    # Get the actual rowid from autobiographical_memory.
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT rowid FROM autobiographical_memory WHERE session_id = 'session-vec' LIMIT 1"
        ).fetchone()
    assert row is not None
    rowid = row[0]

    dims = 768
    # Insert a simple all-zeros embedding, then search with the same vector.
    zero_vec = struct.pack(f"{dims}f", *([0.0] * dims))
    inserted = _db.upsert_message_embeddings_batch({rowid: zero_vec}, tmp_path)
    assert inserted == 1

    results = _db.semantic_search_messages(zero_vec, k=5, root=tmp_path)
    assert len(results) >= 1
    assert results[0]["session_id"] == "session-vec"

# ---------------------------------------------------------------------------
# get_messages_in_range
# ---------------------------------------------------------------------------

def test_get_messages_in_range_returns_messages_in_window(tmp_path: Path) -> None:
    """Messages whose timestamp falls within the window are returned."""
    _db.init_db(tmp_path)
    _db.upsert_session_messages(
        "session-range",
        [
            {"role": "user", "content": "early message"},
            {"role": "assistant", "content": "middle message"},
            {"role": "user", "content": "late message"},
        ],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    # Stamp rows to known timestamps so we can slice.
    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = 'session-range' ORDER BY message_index"
        ).fetchall()
        assert len(rows) == 3
        conn.execute("UPDATE autobiographical_memory SET timestamp_utc = '2026-05-01T10:00:00' WHERE id = ?", (rows[0][0],))
        conn.execute("UPDATE autobiographical_memory SET timestamp_utc = '2026-05-02T10:00:00' WHERE id = ?", (rows[1][0],))
        conn.execute("UPDATE autobiographical_memory SET timestamp_utc = '2026-05-03T10:00:00' WHERE id = ?", (rows[2][0],))
        conn.commit()

    results = _db.get_messages_in_range("2026-05-02T00:00:00", "2026-05-02T23:59:59", root=tmp_path)
    assert len(results) == 1
    assert results[0]["raw_content"] == "middle message"


def test_get_messages_in_range_session_filter(tmp_path: Path) -> None:
    """session_id filter restricts results to the named session."""
    _db.init_db(tmp_path)
    for sid in ("session-a", "session-b"):
        _db.upsert_session_messages(
            sid,
            [{"role": "user", "content": f"message from {sid}"}],
            "global",
            _db.ACTOR_MEMORY_WRITER,
            tmp_path,
        )
    with _db.connect(tmp_path) as conn:
        conn.execute("UPDATE autobiographical_memory SET timestamp_utc = '2026-05-10T12:00:00'")
        conn.commit()

    results = _db.get_messages_in_range(
        "2026-05-10T00:00:00", "2026-05-10T23:59:59",
        session_id="session-a",
        root=tmp_path,
    )
    assert len(results) == 1
    assert results[0]["session_id"] == "session-a"


def test_get_messages_in_range_empty_when_outside_window(tmp_path: Path) -> None:
    """No results when no messages fall in the given window."""
    _db.init_db(tmp_path)
    _db.upsert_session_messages(
        "session-empty",
        [{"role": "user", "content": "outside"}],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    results = _db.get_messages_in_range("2025-01-01T00:00:00", "2025-01-02T00:00:00", root=tmp_path)
    assert results == []


# ---------------------------------------------------------------------------
# get_message_by_id
# ---------------------------------------------------------------------------

def test_get_message_by_id_returns_message(tmp_path: Path) -> None:
    """get_message_by_id returns the row for a valid id."""
    _db.init_db(tmp_path)
    _db.upsert_session_messages(
        "session-exact",
        [{"role": "user", "content": "the exact message"}],
        "global",
        _db.ACTOR_MEMORY_WRITER,
        tmp_path,
    )
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = 'session-exact' LIMIT 1"
        ).fetchone()
    assert row is not None
    msg_id = row[0]

    result = _db.get_message_by_id(msg_id, tmp_path)
    assert result is not None
    assert result["id"] == msg_id
    assert result["raw_content"] == "the exact message"


def test_get_message_by_id_returns_none_for_unknown(tmp_path: Path) -> None:
    """get_message_by_id returns None for an id that does not exist."""
    _db.init_db(tmp_path)
    result = _db.get_message_by_id("nonexistent-id-000", tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# memory_actions
# ---------------------------------------------------------------------------

def test_enqueue_and_claim_memory_action_round_trip(tmp_path: Path) -> None:
    subject = _subjects.create_subject("MA Topic", "desc", None, tmp_path)
    action_id = _db.enqueue_memory_action(
        action_type="finalize_subject",
        subject_id=subject["id"],
        payload={"subject_name": subject["name"]},
        root=tmp_path,
    )
    assert action_id > 0

    claimed = _db.claim_next_memory_action(root=tmp_path)
    assert claimed is not None
    assert claimed["id"] == action_id
    assert claimed["action_type"] == "finalize_subject"
    assert claimed["subject_id"] == subject["id"]
    assert claimed["payload"] == {"subject_name": subject["name"]}
    assert claimed["attempts"] == 1

    rows = _db.peek_pending_memory_actions(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "running"

    _db.complete_memory_action(action_id, root=tmp_path)
    rows = _db.peek_pending_memory_actions(tmp_path)
    assert rows == []


def test_claim_memory_action_returns_none_when_empty(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    assert _db.claim_next_memory_action(root=tmp_path) is None


def test_claim_memory_action_orders_by_id(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Order Topic", "", None, tmp_path)
    first = _db.enqueue_memory_action(
        action_type="finalize_subject", subject_id=subject["id"], root=tmp_path
    )
    second = _db.enqueue_memory_action(
        action_type="finalize_subject", subject_id=subject["id"], root=tmp_path
    )
    assert first < second

    claimed_a = _db.claim_next_memory_action(root=tmp_path)
    assert claimed_a is not None and claimed_a["id"] == first
    claimed_b = _db.claim_next_memory_action(root=tmp_path)
    assert claimed_b is not None and claimed_b["id"] == second
    assert _db.claim_next_memory_action(root=tmp_path) is None


def test_fail_memory_action_retries_then_marks_failed(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Retry Topic", "", None, tmp_path)
    action_id = _db.enqueue_memory_action(
        action_type="finalize_subject", subject_id=subject["id"], root=tmp_path
    )

    # First 4 failures should reset to pending (attempts < MAX_ATTEMPTS=5).
    for expected_attempts in range(1, _db.MEMORY_ACTION_MAX_ATTEMPTS):
        claimed = _db.claim_next_memory_action(root=tmp_path)
        assert claimed is not None and claimed["attempts"] == expected_attempts
        new_status = _db.fail_memory_action(action_id, f"boom {expected_attempts}", root=tmp_path)
        assert new_status == _db.MEMORY_ACTION_STATUS_PENDING

    # 5th claim/fail should mark the row failed permanently.
    claimed = _db.claim_next_memory_action(root=tmp_path)
    assert claimed is not None and claimed["attempts"] == _db.MEMORY_ACTION_MAX_ATTEMPTS
    new_status = _db.fail_memory_action(action_id, "final boom", root=tmp_path)
    assert new_status == _db.MEMORY_ACTION_STATUS_FAILED

    # No more pending rows to claim; failed row is retained.
    assert _db.claim_next_memory_action(root=tmp_path) is None
    rows = _db.peek_pending_memory_actions(tmp_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["last_error"] == "final boom"
    assert rows[0]["attempts"] == _db.MEMORY_ACTION_MAX_ATTEMPTS


def test_claim_memory_action_type_filter(tmp_path: Path) -> None:
    """claim_next_memory_action with action_types only returns matching rows."""
    _db.init_db(tmp_path)
    reconcile_id = _db.enqueue_memory_action(action_type="reconcile_link", subject_id=None, root=tmp_path)
    classify_id = _db.enqueue_memory_action(action_type="classify_message", subject_id=None, root=tmp_path)
    assert reconcile_id < classify_id  # reconcile is older

    # Classify daemon - should skip the older reconcile_link and claim classify_message.
    claimed = _db.claim_next_memory_action(
        root=tmp_path, action_types=["classify_message"]
    )
    assert claimed is not None
    assert claimed["id"] == classify_id
    assert claimed["action_type"] == "classify_message"

    # reconcile_link is still pending - not touched by classify daemon.
    pending = _db.peek_pending_memory_actions(tmp_path)
    pending_ids = [r["id"] for r in pending if r["status"] == "pending"]
    assert reconcile_id in pending_ids

    # Unfiltered claim (no action_types) picks up the remaining reconcile_link.
    claimed2 = _db.claim_next_memory_action(root=tmp_path)
    assert claimed2 is not None
    assert claimed2["id"] == reconcile_id


def test_init_db_resets_stale_running_rows(tmp_path: Path) -> None:
    """A 'running' row left behind by a crashed prior cycle should be reset to pending on init_db."""
    subject = _subjects.create_subject("Stale Topic", "", None, tmp_path)
    action_id = _db.enqueue_memory_action(
        action_type="finalize_subject", subject_id=subject["id"], root=tmp_path
    )
    claimed = _db.claim_next_memory_action(root=tmp_path)
    assert claimed is not None and claimed["id"] == action_id

    # Simulate a server restart: clear the per-process sweep cache so the
    # next init_db() runs the stale-running recovery again.
    _db._stale_running_swept.clear()
    _db.init_db(tmp_path)

    rows = _db.peek_pending_memory_actions(tmp_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"


def test_memory_actions_audit_logged(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Audit Action", "", None, tmp_path)
    action_id = _db.enqueue_memory_action(
        action_type="finalize_subject", subject_id=subject["id"], root=tmp_path
    )
    with _db.connect(tmp_path) as conn:
        rows = conn.execute(
            "SELECT operation FROM audit_log WHERE table_name = 'memory_actions' AND record_id = ? ORDER BY id",
            (str(action_id),),
        ).fetchall()
    assert [r["operation"] for r in rows] == ["INSERT"]


# has_pending_memory_action
# ---------------------------------------------------------------------------

def test_has_pending_memory_action_returns_false_when_empty(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    assert _db.has_pending_memory_action("split_subject", "no-such-id", tmp_path) is False


def test_has_pending_memory_action_returns_true_for_pending(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Split Candidate", "", None, tmp_path)
    _db.enqueue_memory_action(
        action_type="split_subject",
        subject_id=subject["id"],
        payload={"subject_name": "Split Candidate"},
        root=tmp_path,
    )
    assert _db.has_pending_memory_action("split_subject", subject["id"], tmp_path) is True


def test_has_pending_memory_action_returns_true_for_running(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Running Split", "", None, tmp_path)
    _db.enqueue_memory_action(
        action_type="split_subject",
        subject_id=subject["id"],
        root=tmp_path,
    )
    # Claim the action to put it in 'running' state.
    claimed = _db.claim_next_memory_action(root=tmp_path)
    assert claimed is not None
    # Should still return True while the row is 'running'.
    assert _db.has_pending_memory_action("split_subject", subject["id"], tmp_path) is True


def test_has_pending_memory_action_returns_false_for_failed(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Failed Split", "", None, tmp_path)
    action_id = _db.enqueue_memory_action(
        action_type="split_subject",
        subject_id=subject["id"],
        root=tmp_path,
    )
    # Exhaust retries to get a 'failed' row.
    for _ in range(_db.MEMORY_ACTION_MAX_ATTEMPTS):
        _db.claim_next_memory_action(root=tmp_path)
        _db.fail_memory_action(action_id, "forced failure", root=tmp_path)
    rows = _db.peek_pending_memory_actions(tmp_path)
    assert any(r["status"] == "failed" for r in rows)
    # A failed row must not block a fresh enqueue.
    assert _db.has_pending_memory_action("split_subject", subject["id"], tmp_path) is False


def test_has_pending_memory_action_ignores_other_action_types(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Type Check", "", None, tmp_path)
    _db.enqueue_memory_action(
        action_type="finalize_subject",
        subject_id=subject["id"],
        root=tmp_path,
    )
    # finalize_subject pending must not affect the check for split_subject.
    assert _db.has_pending_memory_action("split_subject", subject["id"], tmp_path) is False


# ---------------------------------------------------------------------------
# get_all_messages_for_session
# ---------------------------------------------------------------------------

def test_get_all_messages_for_session_returns_rows_in_index_order(tmp_path: Path) -> None:
    """Returns every message for the session with role/content keys, ordered by index."""
    _db.init_db(tmp_path)
    msgs = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "second message"},
        {"role": "user", "content": "third message"},
    ]
    _db.upsert_session_messages("sess-order", msgs, "global", _db.ACTOR_MEMORY_WRITER, tmp_path)

    rows = _db.get_all_messages_for_session("sess-order", tmp_path)

    assert len(rows) == 3
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "first message"
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content"] == "second message"
    assert rows[2]["role"] == "user"
    assert rows[2]["content"] == "third message"
    # Indices must be strictly ascending.
    indices = [r["message_index"] for r in rows]
    assert indices == sorted(indices)


def test_get_all_messages_for_session_returns_empty_for_unknown_session(tmp_path: Path) -> None:
    """Returns an empty list when the session_id does not exist."""
    _db.init_db(tmp_path)
    rows = _db.get_all_messages_for_session("nonexistent-session-xyz", tmp_path)
    assert rows == []