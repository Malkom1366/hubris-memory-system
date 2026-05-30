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
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str, cfg: dict[str, Any]) -> bytes | None:
    """
    Embed text using the configured embed_model via the active backend adapter.

    Returns the embedding as serialized float32 bytes suitable for direct
    insertion into a vec0 virtual table, or None on any error.

    Errors are logged at DEBUG level so a missing or unloaded model is silent
    and does not interrupt the watcher pipeline.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return None
    model = cfg.get("embed_model", "").strip()
    if not model:
        return None
    return build_backend_adapter(cfg).embed(text, model)
