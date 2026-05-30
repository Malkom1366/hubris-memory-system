"""Tests for the backend adapter layer.

Covers OllamaAdapter.complete(), OllamaAdapter.embed(), the BackendAdapter
protocol, and the build_backend_adapter() factory.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend_adapter_ollama import OllamaAdapter
from backend_adapters import BackendAdapter, build_backend_adapter

# ---------------------------------------------------------------------------
# Test config fixtures
# ---------------------------------------------------------------------------

_CFG = {
    "backend_adapter": "ollama",
    "backends": {
        "ollama": {
            "endpoint": "http://localhost:11434",
            "api_key": "",
            "timeout": 600.0,
        }
    },
}

_CFG_WITH_KEY = {
    "backend_adapter": "ollama",
    "backends": {
        "ollama": {
            "endpoint": "http://localhost:11434",
            "api_key": "sk-secret",
            "timeout": 600.0,
        }
    },
}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_ollama_adapter_satisfies_backend_adapter_protocol() -> None:
    adapter = OllamaAdapter()
    assert isinstance(adapter, BackendAdapter)


# ---------------------------------------------------------------------------
# build_backend_adapter factory
# ---------------------------------------------------------------------------

def test_build_backend_adapter_returns_ollama_adapter() -> None:
    adapter = build_backend_adapter(_CFG)
    assert isinstance(adapter, OllamaAdapter)


def test_build_backend_adapter_uses_ollama_default_when_key_absent() -> None:
    adapter = build_backend_adapter({})
    assert isinstance(adapter, OllamaAdapter)


def test_build_backend_adapter_reads_endpoint_from_backends() -> None:
    cfg = {
        "backend_adapter": "ollama",
        "backends": {"ollama": {"endpoint": "http://custom:9999", "api_key": ""}},
    }
    adapter = build_backend_adapter(cfg)
    assert adapter._endpoint == "http://custom:9999"


def test_build_backend_adapter_raises_on_unknown_name() -> None:
    cfg = {**_CFG, "backend_adapter": "nonexistent"}
    with pytest.raises(ValueError, match="nonexistent"):
        build_backend_adapter(cfg)


# ---------------------------------------------------------------------------
# Helpers for SSE stream mocking
# ---------------------------------------------------------------------------

class _MockStreamCtx:
    """Context manager that simulates an httpx streaming response."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _sse_lines(*pieces: str, extra: list[dict] | None = None) -> list[str]:
    """Build SSE lines that yield the given content pieces."""
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}"
        for p in pieces
    ]
    for chunk in extra or []:
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


# ---------------------------------------------------------------------------
# OllamaAdapter.complete() - happy path
# ---------------------------------------------------------------------------

def test_complete_returns_assembled_content() -> None:
    adapter = OllamaAdapter()
    ctx = _MockStreamCtx(_sse_lines("hello", " world"))
    with patch("httpx.stream", return_value=ctx):
        result = adapter.complete("sys", "usr", "test-model")
    assert result == "hello world"


def test_complete_passes_model_name_in_payload() -> None:
    adapter = OllamaAdapter()
    ctx = _MockStreamCtx(_sse_lines("ok"))
    with patch("httpx.stream", return_value=ctx) as mock_stream:
        adapter.complete("sys", "usr", "my-model")
    assert mock_stream.call_args.kwargs["json"]["model"] == "my-model"


def test_complete_posts_to_chat_completions_endpoint() -> None:
    adapter = OllamaAdapter(endpoint="http://host:1234")
    ctx = _MockStreamCtx(_sse_lines("ok"))
    with patch("httpx.stream", return_value=ctx) as mock_stream:
        adapter.complete("sys", "usr", "model")
    url = mock_stream.call_args.args[1]
    assert url == "http://host:1234/v1/chat/completions"


# ---------------------------------------------------------------------------
# OllamaAdapter.complete() - stripping
# ---------------------------------------------------------------------------

def test_complete_strips_think_blocks() -> None:
    raw = "<think>internal reasoning</think>actual answer"
    ctx = _MockStreamCtx(_sse_lines(raw))
    adapter = OllamaAdapter()
    with patch("httpx.stream", return_value=ctx):
        result = adapter.complete("sys", "usr", "model")
    assert result == "actual answer"
    assert "<think>" not in result


def test_complete_strips_think_block_mid_content() -> None:
    raw = "part one<think>skip</think>part two"
    ctx = _MockStreamCtx(_sse_lines(raw))
    adapter = OllamaAdapter()
    with patch("httpx.stream", return_value=ctx):
        result = adapter.complete("sys", "usr", "model")
    assert result == "part onepart two"


def test_complete_raises_on_zero_content_pieces() -> None:
    ctx = _MockStreamCtx(["data: [DONE]"])
    adapter = OllamaAdapter()
    with patch("httpx.stream", return_value=ctx):
        with pytest.raises(ValueError, match="no content"):
            adapter.complete("sys", "usr", "model")


# ---------------------------------------------------------------------------
# OllamaAdapter.complete() - authentication
# ---------------------------------------------------------------------------

def test_complete_includes_bearer_token_when_api_key_set() -> None:
    adapter = OllamaAdapter(api_key="sk-test")
    ctx = _MockStreamCtx(_sse_lines("hi"))
    with patch("httpx.stream", return_value=ctx) as mock_stream:
        adapter.complete("sys", "usr", "model")
    headers = mock_stream.call_args.kwargs["headers"]
    assert headers.get("Authorization") == "Bearer sk-test"


def test_complete_no_auth_header_when_no_api_key() -> None:
    adapter = OllamaAdapter(api_key="")
    ctx = _MockStreamCtx(_sse_lines("hi"))
    with patch("httpx.stream", return_value=ctx) as mock_stream:
        adapter.complete("sys", "usr", "model")
    headers = mock_stream.call_args.kwargs["headers"]
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# OllamaAdapter.embed() - happy path
# ---------------------------------------------------------------------------

_FAKE_DIM = 1024


def _fake_embed_response(floats: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"data": [{"embedding": floats}]}
    return mock


def test_embed_returns_bytes_on_success() -> None:
    adapter = OllamaAdapter()
    floats = [0.1] * _FAKE_DIM
    with patch("httpx.post", return_value=_fake_embed_response(floats)) as mock_post:
        result = adapter.embed("hello", "mxbai-embed-large")
    assert result is not None
    assert len(result) == _FAKE_DIM * 4
    assert mock_post.call_args.kwargs["json"]["model"] == "mxbai-embed-large"
    assert mock_post.call_args.kwargs["json"]["input"] == "hello"


def test_embed_posts_to_embeddings_endpoint() -> None:
    adapter = OllamaAdapter(endpoint="http://host:5678")
    with patch("httpx.post", return_value=_fake_embed_response([0.0] * _FAKE_DIM)) as mock_post:
        adapter.embed("text", "model")
    assert mock_post.call_args.args[0] == "http://host:5678/v1/embeddings"


# ---------------------------------------------------------------------------
# OllamaAdapter.embed() - error handling
# ---------------------------------------------------------------------------

def test_embed_returns_none_on_http_error() -> None:
    adapter = OllamaAdapter()
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        result = adapter.embed("hello", "model")
    assert result is None


def test_embed_returns_none_on_missing_data_key() -> None:
    adapter = OllamaAdapter()
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {}  # missing "data" key
    with patch("httpx.post", return_value=mock):
        result = adapter.embed("hello", "model")
    assert result is None


# ---------------------------------------------------------------------------
# OllamaAdapter.embed() - authentication
# ---------------------------------------------------------------------------

def test_embed_includes_bearer_token_when_api_key_set() -> None:
    adapter = OllamaAdapter(api_key="sk-secret")
    with patch("httpx.post", return_value=_fake_embed_response([0.0] * _FAKE_DIM)) as mock_post:
        adapter.embed("hello", "model")
    headers = mock_post.call_args.kwargs["headers"]
    assert headers.get("Authorization") == "Bearer sk-secret"


def test_embed_no_auth_header_when_no_api_key() -> None:
    adapter = OllamaAdapter(api_key="")
    with patch("httpx.post", return_value=_fake_embed_response([0.0] * _FAKE_DIM)) as mock_post:
        adapter.embed("hello", "model")
    headers = mock_post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# OllamaAdapter.list_models() - happy path
# ---------------------------------------------------------------------------


def _fake_models_response(names: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"data": [{"id": n} for n in names]}
    return mock


def test_list_models_returns_sorted_model_names() -> None:
    adapter = OllamaAdapter()
    with patch("httpx.get", return_value=_fake_models_response(["zz", "aa", "mm"])):
        result = adapter.list_models()
    assert result == ["aa", "mm", "zz"]


def test_list_models_calls_v1_models_endpoint() -> None:
    adapter = OllamaAdapter(endpoint="http://host:9999")
    with patch("httpx.get", return_value=_fake_models_response([])) as mock_get:
        adapter.list_models()
    assert mock_get.call_args.args[0] == "http://host:9999/v1/models"


def test_list_models_returns_empty_on_connection_error() -> None:
    adapter = OllamaAdapter()
    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        result = adapter.list_models()
    assert result == []


def test_list_models_returns_empty_on_http_status_error() -> None:
    adapter = OllamaAdapter()
    mock = MagicMock()
    mock.raise_for_status.side_effect = httpx.HTTPStatusError(
        "403", request=MagicMock(), response=MagicMock(status_code=403)
    )
    with patch("httpx.get", return_value=mock):
        result = adapter.list_models()
    assert result == []


def test_list_models_returns_empty_on_missing_data_key() -> None:
    adapter = OllamaAdapter()
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {}  # no "data" key
    with patch("httpx.get", return_value=mock):
        result = adapter.list_models()
    assert result == []


def test_list_models_skips_entries_missing_id() -> None:
    adapter = OllamaAdapter()
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"data": [{"id": "good"}, {"no_id": True}]}
    with patch("httpx.get", return_value=mock):
        result = adapter.list_models()
    assert result == ["good"]


def test_list_models_passes_auth_header_when_api_key_set() -> None:
    adapter = OllamaAdapter(api_key="sk-key")
    with patch("httpx.get", return_value=_fake_models_response(["m"])) as mock_get:
        adapter.list_models()
    headers = mock_get.call_args.kwargs["headers"]
    assert headers.get("Authorization") == "Bearer sk-key"
