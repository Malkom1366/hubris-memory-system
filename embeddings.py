"""
Optional embedding support for HuBrIS.

Requires:
  - sqlite-vec installed (PyPI: sqlite-vec)
  - An embedding model configured as embed_model in config.json
  - Ollama running with that model pulled

If either requirement is absent, all functions degrade gracefully: is_available()
returns False, embed() returns None, and the rest of HuBrIS runs normally.
"""

import struct
from typing import Any

from backend_adapters import build_backend_adapter
from log import get_logger

_log = get_logger("hubris.embeddings")

# ---------------------------------------------------------------------------
# sqlite-vec availability
# ---------------------------------------------------------------------------

try:
    import sqlite_vec  # type: ignore[import]
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


def is_available() -> bool:
    """True when sqlite-vec is installed in the current environment."""
    return _SQLITE_VEC_AVAILABLE


def load_extension(conn: Any) -> bool:
    """
    Load the sqlite-vec extension into an open sqlite3 connection.

    Returns True on success, False if sqlite-vec is not installed or if
    extension loading is not supported by this Python build.
    Must be called before any vec0 virtual table operations on this connection.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)  # type: ignore[name-defined]
        conn.enable_load_extension(False)
        return True
    except Exception as exc:
        _log.debug("sqlite-vec extension load failed: %s", exc)
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_float32(floats: list[float]) -> bytes:
    """Pack a list of floats into the binary format sqlite-vec expects."""
    return struct.pack(f"{len(floats)}f", *floats)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Last-resort fallbacks used only when a raw cfg dict that bypassed
# config.load() is passed. Under normal operation config.load() computes
# embed_chunk_size and embed_chunk_overlap from embed_context_tokens via
# compute_budgets() and these constants are never reached.
_DEFAULT_CHUNK_SIZE = 5324   # 65% of 2048-token embed context at 4 chars/token
_DEFAULT_CHUNK_OVERLAP = 532  # 10% of default chunk size


def chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping fixed-size char chunks.

    The last chunk may be shorter than chunk_size. Overlap ensures context
    at chunk boundaries is not lost. Returns a single-element list when the
    text fits in one chunk.
    """
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str, cfg: dict[str, Any]) -> bytes | None:
    """
    Embed a single text string, truncated to one chunk if necessary.

    Used for query vectors (reconcile similarity search, recall_vector) where
    only a single embedding is needed. For storing message embeddings use
    embed_chunks() instead so long messages get full coverage.

    Returns the embedding as serialized float32 bytes suitable for a vec0
    KNN query, or None on any error.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return None
    model = cfg.get("embed_model", "").strip()
    if not model:
        return None
    chunk_size: int = int(cfg.get("embed_chunk_size", _DEFAULT_CHUNK_SIZE))
    if len(text) > chunk_size:
        text = text[:chunk_size]
    return build_backend_adapter(cfg).embed(text, model)


def embed_chunks(text: str, cfg: dict[str, Any]) -> list[bytes]:
    """
    Chunk text and embed each piece, returning a list of embedding byte strings.

    Long messages produce multiple embeddings (one per chunk); short messages
    produce exactly one. Returns an empty list when embeddings are unavailable
    or on the first embed failure so the caller can retry the whole message on
    the next cycle.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return []
    model = cfg.get("embed_model", "").strip()
    if not model:
        return []
    chunk_size: int = int(cfg.get("embed_chunk_size", _DEFAULT_CHUNK_SIZE))
    overlap: int = int(cfg.get("embed_chunk_overlap", _DEFAULT_CHUNK_OVERLAP))
    chunks = chunk_text(text, chunk_size, overlap)
    adapter = build_backend_adapter(cfg)
    result: list[bytes] = []
    for chunk in chunks:
        emb = adapter.embed(chunk, model)
        if emb is None:
            return []  # fail fast; caller retries entire message next cycle
        result.append(emb)
    return result
