"""Tests for the split_subject memory action: _maybe_enqueue_split and
_handle_split_subject in daemon_split.py."""

import json
from pathlib import Path
from unittest import mock

import db as _db
import daemon_split
import subjects as _subjects


# ---------------------------------------------------------------------------
# _maybe_enqueue_split
# ---------------------------------------------------------------------------

def test_maybe_enqueue_split_skips_below_threshold(tmp_path: Path) -> None:
    """Subject with message_count below threshold must not be enqueued."""
    subject = _subjects.create_subject("Small Topic", "", None, tmp_path)
    # message_count defaults to 0 for a brand-new subject
    cfg = {"split_subject_threshold": 150}
    enqueued = daemon_split._maybe_enqueue_split(subject, tmp_path, cfg)
    assert enqueued is False
    assert _db.claim_next_memory_action(root=tmp_path) is None


def test_maybe_enqueue_split_skips_when_threshold_disabled(tmp_path: Path) -> None:
    """Threshold of 0 disables automatic splitting entirely."""
    subject = _subjects.create_subject("Big Topic Zero", "", None, tmp_path)
    # Manually bump message_count in the DB so the function can see a high count.
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
            (subject["id"],),
        )
    # Reload so the dict has the updated count.
    reloaded = next(
        s for s in _subjects.load_subjects(tmp_path) if s["id"] == subject["id"]
    )
    cfg = {"split_subject_threshold": 0}
    enqueued = daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg)
    assert enqueued is False


def test_maybe_enqueue_split_enqueues_when_over_threshold(tmp_path: Path) -> None:
    """Subject at or above threshold must produce a split_subject action."""
    subject = _subjects.create_subject("Fat Topic", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 150 WHERE id = ?",
            (subject["id"],),
        )
    reloaded = next(
        s for s in _subjects.load_subjects(tmp_path) if s["id"] == subject["id"]
    )
    cfg = {"split_subject_threshold": 150}
    enqueued = daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg)
    claimed = _db.claim_next_memory_action(root=tmp_path)
    assert claimed is not None
    assert claimed["action_type"] == "split_subject"
    assert claimed["subject_id"] == subject["id"]
    assert claimed["payload"]["subject_name"] == "Fat Topic"
    assert claimed["payload"]["_split_depth"] == 0


def test_maybe_enqueue_split_skips_when_action_already_pending(tmp_path: Path) -> None:
    """No duplicate enqueue when a split_subject is already pending or running."""
    subject = _subjects.create_subject("Dup Topic", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
            (subject["id"],),
        )
    reloaded = next(
        s for s in _subjects.load_subjects(tmp_path) if s["id"] == subject["id"]
    )
    cfg = {"split_subject_threshold": 150}
    first = daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg)
    second = daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg)
    assert first is True
    assert second is False
    # Only one action should be in the queue.
    rows = _db.peek_pending_memory_actions(tmp_path)
    assert sum(1 for r in rows if r["action_type"] == "split_subject") == 1


def test_maybe_enqueue_split_skips_archived_subjects(tmp_path: Path) -> None:
    """Archived subjects must never be enqueued for splitting."""
    subject = _subjects.create_subject("Archived Topic", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
            (subject["id"],),
        )
    _subjects.set_subject_state(subject["id"], "archived", tmp_path)
    reloaded = next(
        s for s in _subjects.load_subjects(tmp_path) if s["id"] == subject["id"]
    )
    cfg = {"split_subject_threshold": 150}
    assert daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg) is False


def test_maybe_enqueue_split_embeds_depth_in_payload(tmp_path: Path) -> None:
    """split_depth argument must be recorded in the enqueued action payload."""
    subject = _subjects.create_subject("Deep Topic", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
            (subject["id"],),
        )
    reloaded = next(
        s for s in _subjects.load_subjects(tmp_path) if s["id"] == subject["id"]
    )
    cfg = {"split_subject_threshold": 150}
    daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg, split_depth=1)
    claimed = _db.claim_next_memory_action(root=tmp_path)
    assert claimed is not None
    assert claimed["payload"]["_split_depth"] == 1


# ---------------------------------------------------------------------------
# _handle_split_subject
# ---------------------------------------------------------------------------

def _make_children_spec() -> list[dict]:
    return [
        {"name": "Child Alpha", "view": "Alpha view content."},
        {"name": "Child Beta", "view": "Beta view content."},
    ]


def test_handle_split_subject_creates_children_and_marks_parent_split(
    tmp_path: Path,
) -> None:
    """Handler must create child subjects and mark the parent as 'split'.
    No classify_memory action is enqueued - daemon_reclass picks up split subjects
    via its own scanner."""
    parent = _subjects.create_subject("Parent Topic", "big topic", None, tmp_path)

    cfg = {"split_subject_threshold": 150, "meta_model": "test-model"}
    payload = {"subject_name": "Parent Topic", "_split_depth": 0}

    with mock.patch("daemon_split.meta_agent.split_subject_memory", return_value=_make_children_spec()):
        daemon_split._handle_split_subject(tmp_path, cfg, parent["id"], payload)

    # Both children should exist.
    all_subjects = _subjects.load_subjects(tmp_path)
    child_names = {s["name"] for s in all_subjects}
    assert "Child Alpha" in child_names
    assert "Child Beta" in child_names

    # Parent must be 'split', not 'archived'. daemon_reclass handles the archive step.
    parent_after = next(s for s in all_subjects if s["id"] == parent["id"])
    assert parent_after["state"] == "split"

    # No classify_memory action must be pending - daemon_reclass uses its own scanner.
    assert _db.has_pending_memory_action("classify_memory", parent["id"], tmp_path) is False


def test_handle_split_subject_raises_on_missing_subject_id(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    cfg = {}
    import pytest
    with pytest.raises(ValueError, match="missing subject_id"):
        daemon_split._handle_split_subject(tmp_path, cfg, None, {})


def test_handle_split_subject_raises_on_unknown_subject_id(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    cfg = {}
    import pytest
    with pytest.raises(LookupError):
        daemon_split._handle_split_subject(tmp_path, cfg, "deadbeef" * 4, {})


def test_handle_split_subject_raises_when_llm_returns_no_children(
    tmp_path: Path,
) -> None:
    parent = _subjects.create_subject("Empty Split Parent", "", None, tmp_path)
    cfg = {"meta_model": "test-model"}
    import pytest
    with mock.patch("daemon_split.meta_agent.split_subject_memory", return_value=[]):
        with pytest.raises(RuntimeError, match="no children"):
            daemon_split._handle_split_subject(tmp_path, cfg, parent["id"], {})


def test_handle_split_subject_respects_max_depth(tmp_path: Path) -> None:
    """Handler must not enqueue further splits when _split_depth >= _SPLIT_MAX_DEPTH."""
    parent = _subjects.create_subject("Depth Guard Parent", "", None, tmp_path)
    cfg = {"split_subject_threshold": 1, "meta_model": "test-model"}
    # depth = _SPLIT_MAX_DEPTH means we are already at the limit.
    payload = {"subject_name": "Depth Guard Parent", "_split_depth": daemon_split._SPLIT_MAX_DEPTH}

    with mock.patch(
        "daemon_split.meta_agent.split_subject_memory", return_value=_make_children_spec()
    ):
        daemon_split._handle_split_subject(tmp_path, cfg, parent["id"], payload)

    # Children exist but no split_subject action for them was enqueued.
    all_subjects = _subjects.load_subjects(tmp_path)
    child_ids = [s["id"] for s in all_subjects if s["name"] in {"Child Alpha", "Child Beta"}]
    assert len(child_ids) == 2
    for child_id in child_ids:
        assert _db.has_pending_memory_action("split_subject", child_id, tmp_path) is False


def test_handle_split_subject_does_not_re_split_already_split_parent(
    tmp_path: Path,
) -> None:
    """A second invocation cannot re-split a parent already in 'split' state
    because _maybe_enqueue_split skips split subjects."""
    parent = _subjects.create_subject("Double Split", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
            (parent["id"],),
        )
    # Force state to 'split' as daemon_split would leave it.
    _subjects.set_subject_state(parent["id"], "split", tmp_path)
    reloaded = next(
        s for s in _subjects.load_subjects(tmp_path) if s["id"] == parent["id"]
    )
    cfg = {"split_subject_threshold": 150}
    # _maybe_enqueue_split must refuse to re-enqueue for a 'split' parent.
    enqueued = daemon_split._maybe_enqueue_split(reloaded, tmp_path, cfg)
    assert enqueued is False


# ---------------------------------------------------------------------------
# _scan_oversized_subjects
# ---------------------------------------------------------------------------


def test_scan_oversized_subjects_returns_items(tmp_path: Path) -> None:
    """Scanner returns non-archived subjects whose message_count is at or above the threshold."""
    subject = _subjects.create_subject("Big Subject", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200 WHERE id = ?",
            (subject["id"],),
        )
    with mock.patch("daemon_split._config.load", return_value={"split_subject_threshold": 150}):
        result = daemon_split._scan_oversized_subjects(tmp_path)
    assert any(r["subject_id"] == subject["id"] for r in result)
    for item in result:
        assert "subject_id" in item
        assert "payload" in item
        assert "subject_name" in item["payload"]


def test_scan_oversized_subjects_skips_archived(tmp_path: Path) -> None:
    """Scanner must exclude archived subjects."""
    subject = _subjects.create_subject("Archived Big Subject", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 200, state = 'archived' WHERE id = ?",
            (subject["id"],),
        )
    with mock.patch("daemon_split._config.load", return_value={"split_subject_threshold": 150}):
        result = daemon_split._scan_oversized_subjects(tmp_path)
    assert not any(r["subject_id"] == subject["id"] for r in result)


def test_scan_oversized_subjects_returns_empty_when_threshold_disabled(tmp_path: Path) -> None:
    """Threshold of 0 disables scanning entirely and returns []."""
    _subjects.create_subject("Huge Disabled Subject", "", None, tmp_path)
    with mock.patch("daemon_split._config.load", return_value={"split_subject_threshold": 0}):
        result = daemon_split._scan_oversized_subjects(tmp_path)
    assert result == []


def test_scan_oversized_subjects_returns_empty_below_threshold(tmp_path: Path) -> None:
    """Subjects below threshold must not appear in scanner output."""
    subject = _subjects.create_subject("Small Subject", "", None, tmp_path)
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_subjects SET message_count = 10 WHERE id = ?",
            (subject["id"],),
        )
    with mock.patch("daemon_split._config.load", return_value={"split_subject_threshold": 150}):
        result = daemon_split._scan_oversized_subjects(tmp_path)
    assert not any(r["subject_id"] == subject["id"] for r in result)


def test_split_watch_spec_shape(tmp_path: Path) -> None:
    """_WATCH_SPEC has the required structure for daemon_watcher."""
    from daemon_split import _WATCH_SPEC as ws
    assert len(ws) == 1
    spec = ws[0]
    for key in ("interval_s", "action_type", "actor", "scanner"):
        assert key in spec, f"_WATCH_SPEC missing key: {key}"
    assert spec["action_type"] == "split_subject"
    assert callable(spec["scanner"])
