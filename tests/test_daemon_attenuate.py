"""Tests for daemon_attenuate.py:
  - _handle_attenuate_confidence: normal attenuation (pure exponential decay)
  - _scan_eligible_links: returns eligible links, excludes recent links
  - _WATCH_SPEC: shape and completeness
"""

from pathlib import Path

import db as _db
import subjects as _subjects
from daemon_attenuate import (
    _ATTENUATION_FACTOR,
    _ELIGIBILITY_DAYS,
    _WATCH_SPEC,
    _handle_attenuate_confidence,
    _scan_eligible_links,
)


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_db_attenuation.py)
# ---------------------------------------------------------------------------


def _seed_link(tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.9) -> int:
    _db.upsert_session_messages(
        session_id,
        [{"role": "user", "content": f"msg for {session_id}"}],
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
            "SELECT id FROM semantic_links WHERE subject_id = ? LIMIT 1",
            (subject_id,),
        ).fetchone()
    return int(row["id"])


def _backdate_link(tmp_path: Path, link_id: int, days_ago: int) -> None:
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_links SET updated_date_utc = datetime('now', ? || ' days') WHERE id = ?",
            (f"-{days_ago}", link_id),
        )


def _get_confidence(tmp_path: Path, link_id: int) -> float | None:
    with _db.connect(tmp_path) as conn:
        row = conn.execute(
            "SELECT confidence FROM semantic_links WHERE id = ?", (link_id,)
        ).fetchone()
    return float(row["confidence"]) if row else None


# ---------------------------------------------------------------------------
# _handle_attenuate_confidence: normal path
# ---------------------------------------------------------------------------


def test_handle_applies_attenuation_factor(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-ha-1", subject["id"], confidence=0.8)
    expected = 0.8 * _ATTENUATION_FACTOR

    _handle_attenuate_confidence(
        tmp_path, {}, str(link_id),
        {"link_id": link_id, "message_id": "sess-ha-1:0", "subject_id": subject["id"], "current_confidence": 0.8},
    )

    assert abs(_get_confidence(tmp_path, link_id) - expected) < 1e-9


def test_handle_repeated_attenuation_compounds(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-ha-2", subject["id"], confidence=0.6)

    current = 0.6
    for _ in range(5):
        _handle_attenuate_confidence(
            tmp_path, {}, str(link_id),
            {"link_id": link_id, "message_id": "sess-ha-2:0", "subject_id": subject["id"], "current_confidence": current},
        )
        current = _get_confidence(tmp_path, link_id)

    # After 5 rounds, confidence should be lower than initial
    # (0.6 * 0.975^5 ~ 0.526).
    assert current is not None
    assert current < 0.6


# ---------------------------------------------------------------------------
# _handle_attenuate_confidence: very low confidence (no floor, no prune)
# ---------------------------------------------------------------------------


def test_handle_very_low_confidence_still_written(tmp_path: Path) -> None:
    """A confidence that decays to a very small value should be written back,
    not clamped or deleted."""
    start = 0.010
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-low-1", subject["id"], confidence=start)
    expected = start * _ATTENUATION_FACTOR

    _handle_attenuate_confidence(
        tmp_path, {}, str(link_id),
        {"link_id": link_id, "message_id": "sess-low-1:0", "subject_id": subject["id"], "current_confidence": start},
    )

    confidence = _get_confidence(tmp_path, link_id)
    assert confidence is not None
    assert abs(confidence - expected) < 1e-9


# ---------------------------------------------------------------------------
# _scan_eligible_links
# ---------------------------------------------------------------------------


def test_scan_returns_old_links(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-scan-1", subject["id"], confidence=0.75)
    _backdate_link(tmp_path, link_id, days_ago=_ELIGIBILITY_DAYS + 1)

    items = _scan_eligible_links(tmp_path)

    assert any(item["payload"]["link_id"] == link_id for item in items)


def test_scan_excludes_recent_links(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-scan-2", subject["id"], confidence=0.75)
    # Link was just inserted - should not appear.

    items = _scan_eligible_links(tmp_path)

    assert not any(item["payload"]["link_id"] == link_id for item in items)


def test_scan_item_shape(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-scan-3", subject["id"], confidence=0.65)
    _backdate_link(tmp_path, link_id, days_ago=_ELIGIBILITY_DAYS + 2)

    items = _scan_eligible_links(tmp_path)
    item = next(i for i in items if i["payload"]["link_id"] == link_id)

    # subject_id in the item is str(link_id) - used as the dedup key, not the semantic subject id.
    assert item["subject_id"] == str(link_id)
    payload = item["payload"]
    assert set(payload.keys()) == {"link_id", "message_id", "subject_id", "current_confidence"}
    assert payload["link_id"] == link_id
    assert abs(payload["current_confidence"] - 0.65) < 1e-9


def test_scan_returns_empty_when_no_eligible_links(tmp_path: Path) -> None:
    items = _scan_eligible_links(tmp_path)
    assert items == []


# ---------------------------------------------------------------------------
# _WATCH_SPEC shape
# ---------------------------------------------------------------------------


def test_watch_spec_has_required_keys() -> None:
    required = {"interval_s", "action_type", "actor", "scanner"}
    for spec in _WATCH_SPEC:
        assert required.issubset(set(spec.keys())), f"missing keys in spec: {spec}"


def test_watch_spec_action_type() -> None:
    action_types = [s["action_type"] for s in _WATCH_SPEC]
    assert "attenuate_confidence" in action_types


def test_watch_spec_scanner_is_callable() -> None:
    for spec in _WATCH_SPEC:
        assert callable(spec["scanner"]), f"scanner is not callable in spec: {spec}"


def test_watch_spec_interval_is_positive() -> None:
    for spec in _WATCH_SPEC:
        assert spec["interval_s"] > 0


def test_watch_spec_actor() -> None:
    for spec in _WATCH_SPEC:
        if spec["action_type"] == "attenuate_confidence":
            assert spec["actor"] == _db.ACTOR_CONFIDENCE_ATTENUATOR
