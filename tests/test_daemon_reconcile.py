"""Tests for daemon_reconcile.py.

Covers:
  - _WATCH_SPEC shape and required keys
  - _scan_pending_reconciliations: returns pending links, skips reconciled
  - _handle_reconcile_link: supports boosts confidence
  - _handle_reconcile_link: contradicts reduces confidence
  - _handle_reconcile_link: updates reduces confidence more sharply
  - _handle_reconcile_link: marks new link reconciled after judgment
  - _handle_reconcile_link: marks reconciled when no candidates exist
  - _handle_reconcile_link: only considers older links, skips newer ones
  - _handle_reconcile_link: graceful when embed unavailable
  - _handle_reconcile_link: graceful when message not found
  - _parse_verdicts: parses valid JSON array
  - _parse_verdicts: skips non-dict elements
  - _parse_verdicts: handles empty response
  - _parse_verdicts: handles markdown-wrapped JSON
  - _parse_verdicts: returns [] on invalid JSON
  - confidence clamped to [0.0, 1.0]
"""

import json
from pathlib import Path
from unittest import mock

import pytest

import db as _db
import subjects as _subj
from daemon_reconcile import (
    ACTOR_RECONCILE,
    _WATCH_SPEC,
    _handle_reconcile_link,
    _parse_verdicts,
    _scan_pending_reconciliations,
)


# ---------------------------------------------------------------------------
# Shared config (no real LLM/embedding needed for most tests)
# ---------------------------------------------------------------------------

_CFG: dict = {
    "subagent_model": "test-model",
    "reconcile_candidates_k": 5,
    "reconcile_delta_supports": 0.05,
    "reconcile_delta_contradicts": -0.10,
    "reconcile_delta_updates": -0.20,
}

# A date guaranteed to be in the past so candidates qualify as "older"
_OLD_DATE = "2020-01-01 00:00:00"
# A date guaranteed to be in the future so the new link is "newer"
_NEW_DATE = "2099-01-01 00:00:00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_link(
    tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.8
) -> int:
    """Seed one message and a link to the given subject. Return the link id."""
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
        msg_row = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = ? AND message_index = 0",
            (session_id,),
        ).fetchone()
        row = conn.execute(
            "SELECT id FROM semantic_links WHERE message_id = ? AND subject_id = ?",
            (msg_row["id"], subject_id),
        ).fetchone()
    assert row is not None, f"link not created for session {session_id}"
    return int(row["id"])


def _get_confidence(tmp_path: Path, link_id: int) -> float | None:
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT confidence FROM semantic_links WHERE id = ?", (link_id,)
        ).fetchone()
    return float(row["confidence"]) if row else None


def _get_reconciled_at(tmp_path: Path, link_id: int) -> str | None:
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT reconciled_at FROM semantic_links WHERE id = ?", (link_id,)
        ).fetchone()
    assert row is not None
    return row["reconciled_at"]


def _get_message_id(tmp_path: Path, session_id: str, message_index: int = 0) -> str:
    """Look up the UUID4 assigned to a message by upsert_session_messages."""
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = ? AND message_index = ?",
            (session_id, message_index),
        ).fetchone()
    assert row is not None, f"No message found for {session_id}:{message_index}"
    return str(row["id"])


def _make_candidates(link_id: int, confidence: float = 0.8) -> list[dict]:
    """Build a candidate result list as returned by semantic_search_subject_messages."""
    return [
        {
            "link_id": link_id,
            "link_created_date": _OLD_DATE,
            "link_confidence": confidence,
            "raw_content": "older memory content",
            "speaker": "user",
        }
    ]


# ---------------------------------------------------------------------------
# _WATCH_SPEC
# ---------------------------------------------------------------------------

def test_watch_spec_has_required_keys() -> None:
    assert len(_WATCH_SPEC) == 1
    spec = _WATCH_SPEC[0]
    for key in ("interval_s", "action_type", "actor", "scanner"):
        assert key in spec, f"missing key: {key}"


def test_watch_spec_action_type_is_reconcile_link() -> None:
    assert _WATCH_SPEC[0]["action_type"] == "reconcile_link"


def test_watch_spec_actor_is_reconciler() -> None:
    assert _WATCH_SPEC[0]["actor"] == ACTOR_RECONCILE


def test_watch_spec_scanner_is_callable() -> None:
    assert callable(_WATCH_SPEC[0]["scanner"])


def test_watch_spec_interval_is_positive() -> None:
    assert int(_WATCH_SPEC[0]["interval_s"]) > 0


# ---------------------------------------------------------------------------
# _scan_pending_reconciliations
# ---------------------------------------------------------------------------

def test_scan_returns_pending_links(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Scan Test", "desc", None, tmp_path)
    _seed_link(tmp_path, "sess-scan-1", subj["id"])

    items = _scan_pending_reconciliations(tmp_path)

    assert len(items) >= 1
    item = items[0]
    assert "subject_id" in item
    assert "payload" in item
    payload = item["payload"]
    for key in ("link_id", "message_id", "subject_id", "created_date_utc"):
        assert key in payload, f"payload missing key: {key}"


def test_scan_skips_already_reconciled(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj_a = _subj.create_subject("Scan Skip A", "desc", None, tmp_path)
    subj_b = _subj.create_subject("Scan Skip B", "desc", None, tmp_path)

    link_a = _seed_link(tmp_path, "sess-skip-a", subj_a["id"])
    link_b = _seed_link(tmp_path, "sess-skip-b", subj_b["id"])

    # Mark link_b reconciled
    _db.mark_link_reconciled(link_b, ACTOR_RECONCILE, tmp_path)

    items = _scan_pending_reconciliations(tmp_path)
    item_link_ids = {int(i["payload"]["link_id"]) for i in items}

    assert link_a in item_link_ids
    assert link_b not in item_link_ids


def test_scan_dedup_key_is_link_id(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Dedup Key", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-dedup", subj["id"])

    items = _scan_pending_reconciliations(tmp_path)
    matching = [i for i in items if int(i["payload"]["link_id"]) == link_id]
    assert len(matching) == 1
    assert matching[0]["subject_id"] == str(link_id)


def test_scan_empty_when_no_links(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    assert _scan_pending_reconciliations(tmp_path) == []


# ---------------------------------------------------------------------------
# _parse_verdicts
# ---------------------------------------------------------------------------

def test_parse_verdicts_valid_array() -> None:
    raw = json.dumps([
        {"link_id": 1, "relation": "supports"},
        {"link_id": 2, "relation": "contradicts"},
    ])
    result = _parse_verdicts(raw)
    assert len(result) == 2
    assert result[0]["relation"] == "supports"
    assert result[1]["relation"] == "contradicts"


def test_parse_verdicts_empty_array() -> None:
    assert _parse_verdicts("[]") == []


def test_parse_verdicts_empty_string() -> None:
    assert _parse_verdicts("") == []


def test_parse_verdicts_invalid_json_returns_empty() -> None:
    assert _parse_verdicts("not json {") == []


def test_parse_verdicts_non_array_returns_empty() -> None:
    assert _parse_verdicts('{"link_id": 1, "relation": "supports"}') == []


def test_parse_verdicts_skips_non_dict_elements() -> None:
    raw = json.dumps([{"link_id": 1, "relation": "supports"}, "oops", 42])
    result = _parse_verdicts(raw)
    assert len(result) == 1
    assert result[0]["link_id"] == 1


def test_parse_verdicts_strips_markdown_fences() -> None:
    raw = "```json\n" + json.dumps([{"link_id": 3, "relation": "updates"}]) + "\n```"
    result = _parse_verdicts(raw)
    assert len(result) == 1
    assert result[0]["relation"] == "updates"


# ---------------------------------------------------------------------------
# _handle_reconcile_link: core behavior
# ---------------------------------------------------------------------------

def test_handle_reconcile_link_supports_boosts_confidence(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Supports Test", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-supp-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-supp-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-supp-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake-embedding"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "supports"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    expected = 0.8 + 0.05
    assert abs(_get_confidence(tmp_path, old_link_id) - expected) < 1e-9
    assert _get_reconciled_at(tmp_path, new_link_id) is not None


def test_handle_reconcile_link_contradicts_reduces_confidence(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Contradicts Test", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-con-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-con-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-con-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "contradicts"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    expected = 0.8 + (-0.10)
    assert abs(_get_confidence(tmp_path, old_link_id) - expected) < 1e-9
    assert _get_reconciled_at(tmp_path, new_link_id) is not None


def test_handle_reconcile_link_updates_reduces_more_sharply(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Updates Test", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-upd-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-upd-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-upd-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "updates"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    expected = 0.8 + (-0.20)
    assert abs(_get_confidence(tmp_path, old_link_id) - expected) < 1e-9
    assert _get_reconciled_at(tmp_path, new_link_id) is not None


def test_handle_reconcile_link_neutral_verdict_leaves_confidence_unchanged(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Neutral Test", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-neu-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-neu-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-neu-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        # LLM returns no verdicts (neutral)
        mock_bba.return_value.complete.return_value = "[]"
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    # Confidence unchanged
    assert abs(_get_confidence(tmp_path, old_link_id) - 0.8) < 1e-9
    # New link still reconciled
    assert _get_reconciled_at(tmp_path, new_link_id) is not None


# ---------------------------------------------------------------------------
# _handle_reconcile_link: no-candidate paths
# ---------------------------------------------------------------------------

def test_handle_reconcile_link_marks_reconciled_with_no_candidates(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("No Cands", "desc", None, tmp_path)
    new_link_id = _seed_link(tmp_path, "sess-no-cands", subj["id"])

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-no-cands"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=[]),
    ):
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    assert _get_reconciled_at(tmp_path, new_link_id) is not None


def test_handle_reconcile_link_skips_newer_candidates(tmp_path: Path) -> None:
    """Candidates whose link_created_date >= new link's date are filtered out."""
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Newer Filter", "desc", None, tmp_path)
    # Both links are created around "now"; we set a very old new_link date so
    # all real links look newer and should be excluded.
    old_link_id = _seed_link(tmp_path, "sess-newer-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-newer-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-newer-new"),
        "subject_id": subj["id"],
        # Use a past date so all candidates look "newer" and are filtered out
        "created_date_utc": "2000-01-01 00:00:00",
    }
    # Candidate has a link_created_date AFTER the new link's date -> should be skipped
    candidates = [
        {
            "link_id": old_link_id,
            "link_created_date": "2025-01-01 00:00:00",
            "link_confidence": 0.8,
            "raw_content": "newer content",
            "speaker": "user",
        }
    ]

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
    ):
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    # No verdict applied so confidence unchanged
    assert abs(_get_confidence(tmp_path, old_link_id) - 0.8) < 1e-9
    # New link still gets marked reconciled (no older neighbors found)
    assert _get_reconciled_at(tmp_path, new_link_id) is not None


# ---------------------------------------------------------------------------
# _handle_reconcile_link: missing message
# ---------------------------------------------------------------------------

def test_handle_reconcile_link_graceful_when_message_not_found(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("No Msg", "desc", None, tmp_path)
    new_link_id = _seed_link(tmp_path, "sess-no-msg", subj["id"])

    # Payload with a message_id that does not exist
    payload = {
        "link_id": new_link_id,
        "message_id": "does-not-exist:0",
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }

    # No mocks needed; the handler should catch the missing message and mark reconciled
    _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)
    assert _get_reconciled_at(tmp_path, new_link_id) is not None


# ---------------------------------------------------------------------------
# _handle_reconcile_link: embed unavailable
# ---------------------------------------------------------------------------

def test_handle_reconcile_link_skips_when_embed_unavailable(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("No Embed", "desc", None, tmp_path)
    new_link_id = _seed_link(tmp_path, "sess-no-emb", subj["id"])

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-no-emb"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }

    with mock.patch("daemon_reconcile._emb.is_available", return_value=False):
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    # embed unavailable -> graceful skip, NOT reconciled (will retry next pass)
    assert _get_reconciled_at(tmp_path, new_link_id) is None


# ---------------------------------------------------------------------------
# Confidence clamping
# ---------------------------------------------------------------------------

def test_confidence_not_boosted_above_one(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Clamp High", "desc", None, tmp_path)
    # Start near ceiling; supports delta would push above 1.0 without clamping
    old_link_id = _seed_link(tmp_path, "sess-clamp-hi-old", subj["id"], confidence=0.99)
    new_link_id = _seed_link(tmp_path, "sess-clamp-hi-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-clamp-hi-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.99)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "supports"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    result = _get_confidence(tmp_path, old_link_id)
    assert result is not None
    assert result <= 1.0


def test_confidence_not_dropped_below_zero(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Clamp Low", "desc", None, tmp_path)
    # Start near floor; updates delta would push below 0.0 without clamping
    old_link_id = _seed_link(tmp_path, "sess-clamp-lo-old", subj["id"], confidence=0.05)
    new_link_id = _seed_link(tmp_path, "sess-clamp-lo-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-clamp-lo-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    # Large negative delta configured explicitly
    cfg = dict(_CFG, reconcile_delta_updates=-0.90)
    candidates = _make_candidates(old_link_id, confidence=0.05)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "updates"}]
        )
        _handle_reconcile_link(tmp_path, cfg, str(new_link_id), payload)

    result = _get_confidence(tmp_path, old_link_id)
    assert result is not None
    assert result >= 0.0


# ---------------------------------------------------------------------------
# _handle_reconcile_link: subject_relations insertion
# ---------------------------------------------------------------------------

def _count_relations(tmp_path: Path) -> int:
    with _db.connect(tmp_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM subject_relations").fetchone()[0]


def test_handle_reconcile_link_supports_inserts_relation_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Rel Supports", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-rs-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-rs-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-rs-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "supports"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    assert _count_relations(tmp_path) == 1
    rows = _db.get_subject_relations_for_subject(subj["id"], tmp_path)
    assert rows[0]["from_link_id"] == old_link_id
    assert rows[0]["to_link_id"] == new_link_id
    assert rows[0]["relation"] == "supports"


def test_handle_reconcile_link_contradicts_inserts_relation_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Rel Contradicts", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-rc-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-rc-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-rc-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "contradicts"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    assert _count_relations(tmp_path) == 1
    rows = _db.get_subject_relations_for_subject(subj["id"], tmp_path)
    assert rows[0]["relation"] == "contradicts"


def test_handle_reconcile_link_updates_inserts_relation_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Rel Updates", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-ru-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-ru-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-ru-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = json.dumps(
            [{"link_id": old_link_id, "relation": "updates"}]
        )
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    assert _count_relations(tmp_path) == 1
    rows = _db.get_subject_relations_for_subject(subj["id"], tmp_path)
    assert rows[0]["relation"] == "updates"


def test_handle_reconcile_link_neutral_does_not_insert_relation_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Rel Neutral", "desc", None, tmp_path)
    old_link_id = _seed_link(tmp_path, "sess-rn-old", subj["id"], confidence=0.8)
    new_link_id = _seed_link(tmp_path, "sess-rn-new", subj["id"], confidence=0.9)

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-rn-new"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }
    candidates = _make_candidates(old_link_id, confidence=0.8)

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=candidates),
        mock.patch("daemon_reconcile.build_backend_adapter") as mock_bba,
    ):
        mock_bba.return_value.complete.return_value = "[]"
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    assert _count_relations(tmp_path) == 0


def test_handle_reconcile_link_no_candidates_does_not_insert_relation_row(tmp_path: Path) -> None:
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Rel No Cands", "desc", None, tmp_path)
    new_link_id = _seed_link(tmp_path, "sess-rnc", subj["id"])

    payload = {
        "link_id": new_link_id,
        "message_id": _get_message_id(tmp_path, "sess-rnc"),
        "subject_id": subj["id"],
        "created_date_utc": _NEW_DATE,
    }

    with (
        mock.patch("daemon_reconcile._emb.is_available", return_value=True),
        mock.patch("daemon_reconcile._emb.embed", return_value=b"fake"),
        mock.patch("daemon_reconcile._db.semantic_search_subject_messages", return_value=[]),
    ):
        _handle_reconcile_link(tmp_path, _CFG, str(new_link_id), payload)

    assert _count_relations(tmp_path) == 0


# ---------------------------------------------------------------------------
# _scan_pending_reconciliations: vector guard
# ---------------------------------------------------------------------------


def test_scan_returns_empty_when_embeddings_unavailable(tmp_path: Path) -> None:
    """Scanner must short-circuit and return [] when sqlite-vec is absent.
    Without this guard the watcher would enqueue links that the handler
    cannot process, creating an infinite enqueue-skip cycle.
    """
    _db.init_db(tmp_path)
    subj = _subj.create_subject("Guard Subject", "desc", None, tmp_path)
    _seed_link(tmp_path, "sess-guard", subj["id"])

    with mock.patch("daemon_reconcile._emb.is_available", return_value=False):
        result = _scan_pending_reconciliations(tmp_path)

    assert result == []
