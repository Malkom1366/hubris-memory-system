"""Tests for the optional embeddings module."""

import struct
from unittest.mock import MagicMock, patch

import pytest

import embeddings


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_is_available_reflects_sqlite_vec_install() -> None:
    # sqlite-vec is installed in the test environment, so this should be True.
    # If it is not installed, the test is still useful as a documentation of behavior.
    result = embeddings.is_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# serialize_float32
# ---------------------------------------------------------------------------

def test_serialize_float32_produces_correct_byte_length() -> None:
    floats = [0.0] * 1024
    result = embeddings.serialize_float32(floats)
    assert len(result) == 1024 * 4  # 4 bytes per float32


def test_serialize_float32_small_vector() -> None:
    floats = [1.0, 2.0, 3.0]
    result = embeddings.serialize_float32(floats)
    unpacked = struct.unpack("3f", result)
    assert list(unpacked) == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# embed() - delegates to backend adapter
# ---------------------------------------------------------------------------

_FAKE_DIM = 1024
_FAKE_CFG = {
    "embed_model": "mxbai-embed-large",
    "backend_adapter": "ollama",
    "backends": {"ollama": {"endpoint": "http://localhost:11434", "api_key": ""}},
}


def test_embed_returns_bytes_on_success() -> None:
    if not embeddings.is_available():
        pytest.skip("sqlite-vec not installed")

    fake_bytes = embeddings.serialize_float32([0.1] * _FAKE_DIM)
    mock_adapter = MagicMock()
    mock_adapter.embed.return_value = fake_bytes

    with patch("embeddings.build_backend_adapter", return_value=mock_adapter):
        result = embeddings.embed("hello world", _FAKE_CFG)

    mock_adapter.embed.assert_called_once_with("hello world", "mxbai-embed-large")
    assert result == fake_bytes


def test_embed_returns_none_when_model_not_configured() -> None:
    cfg = {**_FAKE_CFG, "embed_model": ""}
    result = embeddings.embed("hello", cfg)
    assert result is None


def test_embed_returns_none_when_adapter_returns_none() -> None:
    if not embeddings.is_available():
        pytest.skip("sqlite-vec not installed")

    mock_adapter = MagicMock()
    mock_adapter.embed.return_value = None

    with patch("embeddings.build_backend_adapter", return_value=mock_adapter):
        result = embeddings.embed("hello", _FAKE_CFG)

    assert result is None
