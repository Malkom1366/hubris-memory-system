"""
Tests for daemon_embed.py - scanners and handlers for message and subject
vector embedding.
"""
import struct
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_DIMS = 1024
_FAKE_BYTES = struct.pack(f"{_DIMS}f", *([0.2] * _DIMS))

_EMBED_CFG = {
    "embed_model": "mxbai-embed-large",
    "backend_adapter": "ollama",
    "backends": {"ollama": {"endpoint": "http://localhost:11434", "api_key": ""}},
    "workspace_id": "global",
}

_NO_MODEL_CFG = {**_EMBED_CFG, "embed_model": ""}


# ---------------------------------------------------------------------------
# Scanner: _scan_messages_needing_embedding
# ---------------------------------------------------------------------------


class TestScanMessagesNeedingEmbedding:
    def test_returns_empty_when_vec_unavailable(self) -> None:
        """Scanner returns [] when sqlite-vec is not installed."""
        with patch("daemon_embed._emb.is_available", return_value=False):
            import daemon_embed
            result = daemon_embed._scan_messages_needing_embedding(root=None)

        assert result == []

    def test_returns_item_for_session_with_new_messages(self) -> None:
        """Session whose total exceeds vectorized HWM produces one item."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={"sess-a": 5}),
            patch("daemon_embed._db.get_session_message_counts", return_value={"sess-a": 10}),
        ):
            import daemon_embed
            result = daemon_embed._scan_messages_needing_embedding(root=None)

        assert len(result) == 1
        assert result[0]["subject_id"] == "sess-a"
        assert result[0]["payload"] == {"session_id": "sess-a"}

    def test_returns_nothing_when_session_fully_vectorized(self) -> None:
        """No item when total equals HWM (nothing new to embed)."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={"sess-b": 10}),
            patch("daemon_embed._db.get_session_message_counts", return_value={"sess-b": 10}),
        ):
            import daemon_embed
            result = daemon_embed._scan_messages_needing_embedding(root=None)

        assert result == []

    def test_returns_nothing_for_empty_db(self) -> None:
        """No sessions in DB produces no items."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={}),
            patch("daemon_embed._db.get_session_message_counts", return_value={}),
        ):
            import daemon_embed
            result = daemon_embed._scan_messages_needing_embedding(root=None)

        assert result == []

    def test_new_session_with_no_hwm_produces_item(self) -> None:
        """Session that has never been vectorized (no HWM entry) produces an item."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={}),
            patch("daemon_embed._db.get_session_message_counts", return_value={"sess-new": 3}),
        ):
            import daemon_embed
            result = daemon_embed._scan_messages_needing_embedding(root=None)

        assert len(result) == 1
        assert result[0]["subject_id"] == "sess-new"


# ---------------------------------------------------------------------------
# Scanner: _scan_subjects_needing_embedding
# ---------------------------------------------------------------------------


class TestScanSubjectsNeedingEmbedding:
    def test_returns_empty_when_vec_unavailable(self) -> None:
        """Scanner returns [] when sqlite-vec is not installed."""
        with patch("daemon_embed._emb.is_available", return_value=False):
            import daemon_embed
            result = daemon_embed._scan_subjects_needing_embedding(root=None)

        assert result == []

    def test_returns_trigger_item_when_subjects_pending(self) -> None:
        """Returns a single item with subject_id=None when subjects need embedding."""
        stale_row = {"id": "subj-1", "memory_content": "Some content."}
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.get_subjects_needing_embedding", return_value=[stale_row]),
        ):
            import daemon_embed
            result = daemon_embed._scan_subjects_needing_embedding(root=None)

        assert len(result) == 1
        assert result[0]["subject_id"] is None
        assert result[0]["payload"] == {}

    def test_returns_empty_when_no_subjects_pending(self) -> None:
        """Returns [] when no subjects need embedding."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.get_subjects_needing_embedding", return_value=[]),
        ):
            import daemon_embed
            result = daemon_embed._scan_subjects_needing_embedding(root=None)

        assert result == []


# ---------------------------------------------------------------------------
# Handler: _handle_embed_subjects
# ---------------------------------------------------------------------------


class TestHandleEmbedSubjects:
    def test_embeds_stale_subject(self) -> None:
        """Handler calls upsert_subject_embedding for each stale subject."""
        stale_row = {"id": "subj-stale", "memory_content": "Synthesized content."}

        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.get_subjects_needing_embedding", return_value=[stale_row]),
            patch("daemon_embed._emb.embed", return_value=_FAKE_BYTES),
            patch("daemon_embed._db.upsert_subject_embedding") as mock_upsert,
        ):
            import daemon_embed
            daemon_embed._handle_embed_subjects(root=None, cfg=_EMBED_CFG, subject_id=None, payload={})

        mock_upsert.assert_called_once_with("subj-stale", _FAKE_BYTES, None)

    def test_skips_when_vec_unavailable(self) -> None:
        """Handler does nothing when sqlite-vec is not installed."""
        with (
            patch("daemon_embed._emb.is_available", return_value=False),
            patch("daemon_embed._db.get_subjects_needing_embedding") as mock_get,
        ):
            import daemon_embed
            daemon_embed._handle_embed_subjects(root=None, cfg=_EMBED_CFG, subject_id=None, payload={})

        mock_get.assert_not_called()

    def test_skips_when_embed_model_not_configured(self) -> None:
        """Handler does nothing when embed_model is blank."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.get_subjects_needing_embedding") as mock_get,
        ):
            import daemon_embed
            daemon_embed._handle_embed_subjects(root=None, cfg=_NO_MODEL_CFG, subject_id=None, payload={})

        mock_get.assert_not_called()

    def test_stops_on_embed_failure(self) -> None:
        """If embed returns None, no upsert is attempted."""
        stale_row = {"id": "subj-fail", "memory_content": "Content."}

        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.get_subjects_needing_embedding", return_value=[stale_row]),
            patch("daemon_embed._emb.embed", return_value=None),
            patch("daemon_embed._db.upsert_subject_embedding") as mock_upsert,
        ):
            import daemon_embed
            daemon_embed._handle_embed_subjects(root=None, cfg=_EMBED_CFG, subject_id=None, payload={})

        mock_upsert.assert_not_called()

    def test_skips_subject_with_empty_content(self) -> None:
        """Subjects with empty memory_content after strip are not embedded."""
        empty_row = {"id": "subj-empty", "memory_content": ""}

        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.get_subjects_needing_embedding", return_value=[empty_row]),
            patch("daemon_embed._emb.embed") as mock_embed,
            patch("daemon_embed._db.upsert_subject_embedding") as mock_upsert,
        ):
            import daemon_embed
            daemon_embed._handle_embed_subjects(root=None, cfg=_EMBED_CFG, subject_id=None, payload={})

        mock_embed.assert_not_called()
        mock_upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Handler: _handle_embed_messages
# ---------------------------------------------------------------------------


class TestHandleEmbedMessages:
    def test_embeds_new_messages_and_advances_hwm(self) -> None:
        """Handler embeds messages past the HWM and saves the updated HWM."""
        rows = [
            {"rowid": 1, "message_index": 5, "raw_content": "Hello world"},
            {"rowid": 2, "message_index": 6, "raw_content": "Second message"},
        ]
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={"sess-x": 5}),
            patch("daemon_embed._db.get_messages_needing_embedding", return_value=rows),
            patch("daemon_embed._emb.embed", return_value=_FAKE_BYTES),
            patch("daemon_embed._db.upsert_message_embeddings_batch") as mock_upsert,
            patch("daemon_embed._db.save_vectorized_counts") as mock_save,
        ):
            import daemon_embed
            daemon_embed._handle_embed_messages(
                root=None,
                cfg=_EMBED_CFG,
                subject_id="sess-x",
                payload={"session_id": "sess-x"},
            )

        mock_upsert.assert_called_once_with({1: _FAKE_BYTES, 2: _FAKE_BYTES}, None)
        saved_counts = mock_save.call_args[0][0]
        assert saved_counts["sess-x"] == 7  # message_index 6 + 1

    def test_skips_when_vec_unavailable(self) -> None:
        """Handler does nothing when sqlite-vec is not installed."""
        with (
            patch("daemon_embed._emb.is_available", return_value=False),
            patch("daemon_embed._db.load_vectorized_counts") as mock_load,
        ):
            import daemon_embed
            daemon_embed._handle_embed_messages(
                root=None, cfg=_EMBED_CFG, subject_id="sess-y", payload={"session_id": "sess-y"}
            )

        mock_load.assert_not_called()

    def test_skips_when_embed_model_not_configured(self) -> None:
        """Handler does nothing when embed_model is blank."""
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts") as mock_load,
        ):
            import daemon_embed
            daemon_embed._handle_embed_messages(
                root=None, cfg=_NO_MODEL_CFG, subject_id="sess-z", payload={"session_id": "sess-z"}
            )

        mock_load.assert_not_called()

    def test_stops_on_embed_failure_without_saving_partial(self) -> None:
        """If embed fails on first message, no upsert and HWM is not advanced."""
        rows = [{"rowid": 10, "message_index": 0, "raw_content": "Fail me"}]
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={}),
            patch("daemon_embed._db.get_messages_needing_embedding", return_value=rows),
            patch("daemon_embed._emb.embed", return_value=None),
            patch("daemon_embed._db.upsert_message_embeddings_batch") as mock_upsert,
            patch("daemon_embed._db.save_vectorized_counts") as mock_save,
        ):
            import daemon_embed
            daemon_embed._handle_embed_messages(
                root=None, cfg=_EMBED_CFG, subject_id="sess-f", payload={"session_id": "sess-f"}
            )

        mock_upsert.assert_not_called()
        mock_save.assert_not_called()

    def test_skips_empty_message_but_advances_hwm(self) -> None:
        """Empty raw_content messages advance the HWM without calling embed."""
        rows = [{"rowid": 20, "message_index": 3, "raw_content": ""}]
        with (
            patch("daemon_embed._emb.is_available", return_value=True),
            patch("daemon_embed._db.load_vectorized_counts", return_value={"sess-e": 3}),
            patch("daemon_embed._db.get_messages_needing_embedding", return_value=rows),
            patch("daemon_embed._emb.embed") as mock_embed,
            patch("daemon_embed._db.upsert_message_embeddings_batch") as mock_upsert,
            patch("daemon_embed._db.save_vectorized_counts") as mock_save,
        ):
            import daemon_embed
            daemon_embed._handle_embed_messages(
                root=None, cfg=_EMBED_CFG, subject_id="sess-e", payload={"session_id": "sess-e"}
            )

        mock_embed.assert_not_called()
        mock_upsert.assert_not_called()
        # HWM should advance past the empty message
        saved_counts = mock_save.call_args[0][0]
        assert saved_counts["sess-e"] == 4
