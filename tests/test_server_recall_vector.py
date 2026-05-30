"""Tests for the recency weighting logic in recall_vector and config defaults."""

import struct
from unittest.mock import MagicMock, patch

import config as _config
import server


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_recency_alpha_default_exists() -> None:
    assert "recency_alpha" in _config.DEFAULTS


def test_recency_lambda_default_exists() -> None:
    assert "recency_lambda" in _config.DEFAULTS


def test_recency_alpha_is_float() -> None:
    assert isinstance(_config.DEFAULTS["recency_alpha"], float)


def test_recency_lambda_is_float() -> None:
    assert isinstance(_config.DEFAULTS["recency_lambda"], float)


def test_recency_alpha_default_is_positive() -> None:
    assert _config.DEFAULTS["recency_alpha"] > 0.0


def test_recency_lambda_default_is_positive() -> None:
    assert _config.DEFAULTS["recency_lambda"] > 0.0


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------

_DIMS = 1024
_FAKE_BYTES = struct.pack(f"{_DIMS}f", *([0.1] * _DIMS))

_BASE_CFG = {
    "embed_model": "mxbai-embed-large",
    "backend_adapter": "ollama",
    "backends": {"ollama": {"endpoint": "http://localhost:11434", "api_key": ""}},
    "workspace_id": "global",
    "max_recall_tokens": 6000,
    "recency_alpha": 0.3,
    "recency_lambda": 0.005,
}


def _fake_row(
    session_id: str,
    msg_id: str,
    distance: float,
    link_created_date: str,
    subject: str = "Test",
) -> dict:
    return {
        "id": msg_id,
        "session_id": session_id,
        "message_index": 0,
        "speaker": "user",
        "raw_content": f"content from {session_id}",
        "timestamp_utc": "2026-01-01T00:00:00",
        "distance": distance,
        "subject_name": subject,
        "link_created_date": link_created_date,
    }


# ---------------------------------------------------------------------------
# recall_vector recency ranking
# ---------------------------------------------------------------------------


def test_recent_link_ranks_above_older_link_with_same_similarity() -> None:
    """A message with a more recent semantic link should rank above one with an older link,
    given the same distance."""
    recent_row = _fake_row("sess-recent", "id-recent", 0.1, "2026-05-29T10:00:00")
    old_row = _fake_row("sess-old", "id-old", 0.1, "2025-01-01T10:00:00")
    # Both have identical distance but very different link dates.

    mock_cfg = dict(_BASE_CFG)

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=mock_cfg),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[old_row, recent_row]),
        patch("server._db.semantic_search_subjects", return_value=[]),
    ):
        result = server.recall_vector(query="test query", k=2)

    # The recent session should appear first (rank 1) in the output.
    recent_idx = result.find("sess-rece"[:8])
    old_idx = result.find("sess-old"[:8])
    assert recent_idx < old_idx, (
        "Recent message should be ranked before older message with same similarity"
    )


def test_alpha_zero_disables_recency_sorting() -> None:
    """With recency_alpha=0.0, results retain their original distance-based order."""
    closer_row = _fake_row("sess-close", "id-close", 0.05, "2025-01-01T00:00:00")
    farther_row = _fake_row("sess-far", "id-far", 0.2, "2026-06-01T00:00:00")
    # closer_row has a smaller distance (more similar) but older link date.
    # With alpha=0, no recency boost - closer should still rank first.

    mock_cfg = dict(_BASE_CFG, recency_alpha=0.0)

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=mock_cfg),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch(
            "server._db.semantic_search_messages",
            return_value=[closer_row, farther_row],
        ),
        patch("server._db.semantic_search_subjects", return_value=[]),
    ):
        result = server.recall_vector(query="test query", k=2)

    # With alpha=0, the original order (closer first) should be preserved because
    # there is no recency re-sort; the input list is already sorted by distance.
    close_idx = result.find("sess-clos"[:8])
    far_idx = result.find("sess-far"[:8])
    assert close_idx < far_idx, (
        "With alpha=0 recency should not change distance-based ranking"
    )


def test_recall_vector_output_uses_score_label() -> None:
    """recall_vector should display 'score=' in each result row, not 'dist='."""
    row = _fake_row("sess-score", "id-score", 0.1, "2026-05-01T00:00:00")

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=dict(_BASE_CFG)),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[row]),
        patch("server._db.semantic_search_subjects", return_value=[]),
    ):
        result = server.recall_vector(query="test query", k=1)

    assert "score=" in result
    assert "dist=" not in result


def test_recall_vector_no_results_returns_error_message() -> None:
    """recall_vector returns the no-results message when both message and subject searches are empty."""
    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=dict(_BASE_CFG)),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[]),
        patch("server._db.semantic_search_subjects", return_value=[]),
    ):
        result = server.recall_vector(query="test query", k=5)

    assert "No similar" in result


# ---------------------------------------------------------------------------
# Subject hits in recall_vector
# ---------------------------------------------------------------------------


def _fake_subject_row(
    subject_id: str,
    name: str,
    dewey_id: str,
    distance: float,
    memory_content: str = "Synthesized content.",
) -> dict:
    return {
        "id": subject_id,
        "name": name,
        "dewey_id": dewey_id,
        "memory_content": memory_content,
        "distance": distance,
    }


def test_recall_vector_includes_subject_hit_in_output() -> None:
    """When semantic_search_subjects returns a row it should appear in the output."""
    subj_row = _fake_subject_row("subj-01", "Authentication", "1.0", 0.05)

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=dict(_BASE_CFG)),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[]),
        patch("server._db.semantic_search_subjects", return_value=[subj_row]),
    ):
        result = server.recall_vector(query="authentication", k=5)

    assert "Authentication" in result


def test_recall_vector_subject_result_type_label() -> None:
    """Subject rows should be labelled type=SUBJECT in the output."""
    subj_row = _fake_subject_row("subj-02", "Billing", "2.0", 0.1)

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=dict(_BASE_CFG)),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[]),
        patch("server._db.semantic_search_subjects", return_value=[subj_row]),
    ):
        result = server.recall_vector(query="billing", k=5)

    assert "type=SUBJECT" in result


def test_recall_vector_message_result_type_label() -> None:
    """Message rows should be labelled type=MESSAGE in the output."""
    msg_row = _fake_row("sess-msg", "id-msg", 0.1, "2026-05-01T00:00:00")

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=dict(_BASE_CFG)),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[msg_row]),
        patch("server._db.semantic_search_subjects", return_value=[]),
    ):
        result = server.recall_vector(query="test", k=1)

    assert "type=MESSAGE" in result


def test_recall_vector_subject_ranks_above_distant_message() -> None:
    """A closer subject hit should appear before a more distant message hit."""
    subj_row = _fake_subject_row("subj-03", "Networking", "3.0", 0.02)
    msg_row = _fake_row("sess-net", "id-net", 0.4, "2026-05-01T00:00:00")

    with (
        patch("server._emb.is_available", return_value=True),
        patch("server._config.load", return_value=dict(_BASE_CFG, recency_alpha=0.0)),
        patch("server._config.memory_root", return_value=None),
        patch("server._emb.embed", return_value=_FAKE_BYTES),
        patch("server._db.semantic_search_messages", return_value=[msg_row]),
        patch("server._db.semantic_search_subjects", return_value=[subj_row]),
    ):
        result = server.recall_vector(query="networking", k=5)

    subj_idx = result.find("Networking")
    msg_idx = result.find("sess-net"[:8])
    assert subj_idx < msg_idx, "Closer subject should rank before more distant message"
