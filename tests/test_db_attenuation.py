"""Tests for the db.py confidence attenuation functions:
  - get_links_eligible_for_attenuation
  - attenuate_link_confidence
  - delete_semantic_link
"""

from pathlib import Path

import db as _db
import subjects as _subjects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_link(tmp_path: Path, session_id: str, subject_id: str, confidence: float = 0.9) -> int:
    """Create a message and a semantic_link; return the link's integer rowid."""
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
    """Move a semantic_link's updated_date_utc back in time for eligibility tests."""
    with _db.connect(tmp_path) as conn:
        conn.execute(
            "UPDATE semantic_links SET updated_date_utc = datetime('now', ? || ' days') WHERE id = ?",
            (f"-{days_ago}", link_id),
        )


# ---------------------------------------------------------------------------
# get_links_eligible_for_attenuation
# ---------------------------------------------------------------------------


def test_returns_links_older_than_cutoff(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-elig-1", subject["id"], confidence=0.9)
    _backdate_link(tmp_path, link_id, days_ago=8)

    eligible = _db.get_links_eligible_for_attenuation(cutoff_days=7, root=tmp_path)

    assert any(item["id"] == link_id for item in eligible)


def test_excludes_links_newer_than_cutoff(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-elig-2", subject["id"], confidence=0.9)
    # Link was just inserted - updated_date_utc is 'now'; should NOT appear.

    eligible = _db.get_links_eligible_for_attenuation(cutoff_days=7, root=tmp_path)

    assert not any(item["id"] == link_id for item in eligible)


def test_returned_row_has_expected_keys(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-elig-3", subject["id"], confidence=0.77)
    _backdate_link(tmp_path, link_id, days_ago=10)

    eligible = _db.get_links_eligible_for_attenuation(cutoff_days=7, root=tmp_path)
    item = next(i for i in eligible if i["id"] == link_id)

    assert set(item.keys()) == {"id", "message_id", "subject_id", "confidence"}
    with _db.connect(tmp_path) as conn:
        _row = conn.execute(
            "SELECT id FROM autobiographical_memory WHERE session_id = 'sess-elig-3' AND message_index = 0",
        ).fetchone()
    assert item["message_id"] == _row["id"]
    assert item["subject_id"] == subject["id"]
    assert abs(item["confidence"] - 0.77) < 1e-9


def test_empty_result_when_no_old_links(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    _seed_link(tmp_path, "sess-elig-4", subject["id"])

    eligible = _db.get_links_eligible_for_attenuation(cutoff_days=7, root=tmp_path)

    assert eligible == []


# ---------------------------------------------------------------------------
# attenuate_link_confidence
# ---------------------------------------------------------------------------


def test_attenuate_writes_new_confidence(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-att-1", subject["id"], confidence=0.8)

    _db.attenuate_link_confidence(link_id, 0.75, _db.ACTOR_CONFIDENCE_ATTENUATOR, tmp_path)

    with _db.connect(tmp_path) as conn:
        row = conn.execute("SELECT confidence FROM semantic_links WHERE id = ?", (link_id,)).fetchone()
    assert abs(row["confidence"] - 0.75) < 1e-9


def test_attenuate_resets_updated_date(tmp_path: Path) -> None:
    """After attenuation, updated_date_utc should be close to now, so the link
    is no longer immediately eligible for another attenuation pass."""
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-att-2", subject["id"])
    _backdate_link(tmp_path, link_id, days_ago=10)

    _db.attenuate_link_confidence(link_id, 0.8, _db.ACTOR_CONFIDENCE_ATTENUATOR, tmp_path)

    # Link should no longer appear in the eligible set.
    eligible = _db.get_links_eligible_for_attenuation(cutoff_days=7, root=tmp_path)
    assert not any(i["id"] == link_id for i in eligible)


def test_attenuate_sets_updated_by(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-att-3", subject["id"])

    _db.attenuate_link_confidence(link_id, 0.7, _db.ACTOR_CONFIDENCE_ATTENUATOR, tmp_path)

    with _db.connect(tmp_path) as conn:
        row = conn.execute("SELECT updated_by FROM semantic_links WHERE id = ?", (link_id,)).fetchone()
    assert row["updated_by"] == _db.ACTOR_CONFIDENCE_ATTENUATOR


# ---------------------------------------------------------------------------
# delete_semantic_link
# ---------------------------------------------------------------------------


def test_delete_removes_link_row(tmp_path: Path) -> None:
    subject = _subjects.create_subject("Topic", "desc", None, tmp_path)
    link_id = _seed_link(tmp_path, "sess-del-1", subject["id"])

    _db.delete_semantic_link(link_id, _db.ACTOR_CONFIDENCE_ATTENUATOR, tmp_path)

    with _db.connect(tmp_path) as conn:
        row = conn.execute("SELECT id FROM semantic_links WHERE id = ?", (link_id,)).fetchone()
    assert row is None


def test_delete_does_not_affect_other_links(tmp_path: Path) -> None:
    subject_a = _subjects.create_subject("Topic A", "desc", None, tmp_path)
    subject_b = _subjects.create_subject("Topic B", "desc", None, tmp_path)
    link_a = _seed_link(tmp_path, "sess-del-2a", subject_a["id"])
    link_b = _seed_link(tmp_path, "sess-del-2b", subject_b["id"])

    _db.delete_semantic_link(link_a, _db.ACTOR_CONFIDENCE_ATTENUATOR, tmp_path)

    with _db.connect(tmp_path) as conn:
        row = conn.execute("SELECT id FROM semantic_links WHERE id = ?", (link_b,)).fetchone()
    assert row is not None


def test_actor_constant_value() -> None:
    """Ensure the constant is present and non-empty."""
    assert _db.ACTOR_CONFIDENCE_ATTENUATOR == "ConfidenceAttenuator"
