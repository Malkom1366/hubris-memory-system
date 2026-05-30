"""Tests for daemon_synthesize.py.

Covers:
  - _WATCH_SPEC shape and required keys
  - _scan_archived_subjects_needing_synthesis: returns unsynthesized archived subjects
  - _scan_archived_subjects_needing_synthesis: skips subjects with synthesized_at set
  - _scan_archived_subjects_needing_synthesis: skips non-archived subjects
  - _handle_finalize_subject: calls mark_subject_synthesized on success
"""

from pathlib import Path
from unittest import mock

import db as _db
import subjects as _subjects
from daemon_synthesize import (
    _WATCH_SPEC,
    _scan_archived_subjects_needing_synthesis,
)


# ---------------------------------------------------------------------------
# _WATCH_SPEC
# ---------------------------------------------------------------------------


def test_synthesize_watch_spec_shape() -> None:
    """_WATCH_SPEC must have the required keys daemon_watcher needs."""
    assert len(_WATCH_SPEC) == 1
    spec = _WATCH_SPEC[0]
    for key in ("interval_s", "action_type", "actor", "scanner"):
        assert key in spec, f"_WATCH_SPEC missing key: {key}"
    assert spec["action_type"] == "finalize_subject"
    assert callable(spec["scanner"])


# ---------------------------------------------------------------------------
# _scan_archived_subjects_needing_synthesis
# ---------------------------------------------------------------------------


def test_scan_returns_unsynthesized_archived_subjects(tmp_path: Path) -> None:
    """Archived subjects with synthesized_at NULL must appear in scanner output."""
    subj = _subjects.create_subject("Archive Me", "", None, tmp_path)
    _subjects.set_subject_state(subj["id"], "archived", tmp_path)

    result = _scan_archived_subjects_needing_synthesis(tmp_path)

    assert any(r["subject_id"] == subj["id"] for r in result)
    for item in result:
        assert "subject_id" in item
        assert "payload" in item
        assert "subject_name" in item["payload"]


def test_scan_skips_synthesized_archived_subjects(tmp_path: Path) -> None:
    """Archived subjects that already have synthesized_at must not be re-queued."""
    subj = _subjects.create_subject("Already Synthesized", "", None, tmp_path)
    _subjects.set_subject_state(subj["id"], "archived", tmp_path)
    _db.mark_subject_synthesized(subj["id"], _db.ACTOR_MEMORY_ACTIONS, tmp_path)

    result = _scan_archived_subjects_needing_synthesis(tmp_path)

    assert not any(r["subject_id"] == subj["id"] for r in result)


def test_scan_skips_non_archived_subjects(tmp_path: Path) -> None:
    """Open and dormant subjects must not appear in the synthesize scanner output."""
    open_subj = _subjects.create_subject("Open Subject", "", None, tmp_path)
    dormant_subj = _subjects.create_subject("Dormant Subject", "", None, tmp_path)
    _subjects.set_subject_state(dormant_subj["id"], "dormant", tmp_path)

    result = _scan_archived_subjects_needing_synthesis(tmp_path)

    ids = {r["subject_id"] for r in result}
    assert open_subj["id"] not in ids
    assert dormant_subj["id"] not in ids


def test_scan_returns_empty_with_no_archived_subjects(tmp_path: Path) -> None:
    """Empty result when no archived subjects exist."""
    _db.init_db(tmp_path)
    assert _scan_archived_subjects_needing_synthesis(tmp_path) == []


# ---------------------------------------------------------------------------
# _handle_finalize_subject: mark_subject_synthesized called on success
# ---------------------------------------------------------------------------


def test_handle_finalize_subject_marks_synthesized(tmp_path: Path) -> None:
    """Successful finalize_subject must stamp synthesized_at so the scanner
    does not re-queue the subject on subsequent passes."""
    from daemon_synthesize import _handle_finalize_subject

    subj = _subjects.create_subject("Finalize Stamp", "", None, tmp_path)
    _subjects.set_subject_state(subj["id"], "archived", tmp_path)
    cfg = {"meta_model": "test-model", "split_subject_threshold": 150}

    with (
        mock.patch(
            "daemon_synthesize.meta_agent.finalize_subject_memory",
            return_value=("synthesized view", "log entry"),
        ),
        mock.patch("daemon_synthesize._subjects.write_subject_memory"),
        mock.patch("daemon_synthesize._subjects.read_subject_memory", return_value=""),
        mock.patch("daemon_synthesize._subjects.parse_memory_view", return_value={}),
        mock.patch("daemon_synthesize._subjects.parse_memory_log", return_value=[]),
        mock.patch("daemon_synthesize._subjects.compose_memory_file", return_value="content"),
        mock.patch("daemon_synthesize._maybe_enqueue_split"),
    ):
        _handle_finalize_subject(tmp_path, cfg, subj["id"], {"subject_name": "Finalize Stamp"})

    # synthesized_at must be set after a successful handler run.
    assert _scan_archived_subjects_needing_synthesis(tmp_path) == [] or all(
        r["subject_id"] != subj["id"]
        for r in _scan_archived_subjects_needing_synthesis(tmp_path)
    )
