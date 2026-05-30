"""
Catalog management for HuBrIS.

The catalog is HuBrIS's Table of Contents - the authoritative index of all known
subjects and their memory locations. It serves two purposes:

  1. Internal state: persisted at ~/.hubris/<workspace_id>/catalog.json
  2. Context anchor: a special marker message injected into the Continue session
     transcript so the agent can see the current subject map on every turn.

ANCHOR INJECTION / TRUNCATION DETECTION
----------------------------------------
HuBrIS injects a catalog anchor as the first message in a session file. On each
watcher cycle, HuBrIS checks whether that anchor is still present. If it is
missing, the agent's context was truncated and HuBrIS must restore it.

Restoration is always two separate writes:
  1. Verbatim catalog anchor restored from catalog.json (IDs and hierarchy must
     be exact - never re-inferred or recomputed from current state).
  2. Compacted summary of the dropped conversational content written as a second
     message so the agent has orientation.

ANCHOR FORMAT
-------------
The anchor message role is "user" so Continue displays it in the context.
The content starts with ANCHOR_MARKER on its own line, followed by the catalog
body as plain text. The marker must be the very first non-whitespace token in
the content so detection is a simple startswith check.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import config as _config

ANCHOR_MARKER = "[HUBRIS-CATALOG-ANCHOR]"
_CATALOG_FILENAME = "catalog.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _catalog_path(root: Path) -> Path:
    return root / _CATALOG_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Catalog load / save
# ---------------------------------------------------------------------------

def load_catalog(root: Path | None = None) -> dict[str, Any]:
    """
    Load the catalog from disk. Returns a dict with at minimum:
        {"subjects": [...], "last_updated": "...", "version": 1}
    Returns an empty catalog if the file does not exist.
    """
    if root is None:
        root = _config.memory_root()
    path = _catalog_path(root)
    if not path.exists():
        return _empty_catalog()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure required fields exist.
    data.setdefault("subjects", [])
    data.setdefault("last_updated", _now_iso())
    data.setdefault("version", 1)
    return data


def save_catalog(catalog: dict[str, Any], root: Path | None = None) -> None:
    """Persist the catalog to disk."""
    if root is None:
        root = _config.memory_root()
    root.mkdir(parents=True, exist_ok=True)
    catalog["last_updated"] = _now_iso()
    with open(_catalog_path(root), "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def _empty_catalog() -> dict[str, Any]:
    return {
        "subjects": [],
        "last_updated": _now_iso(),
        "version": 1,
    }


# ---------------------------------------------------------------------------
# Catalog building from subjects manifest
# ---------------------------------------------------------------------------

def rebuild_catalog_from_subjects(
    subjects: list[dict[str, Any]],
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Rebuild and save the catalog from the current subjects manifest.
    Returns the new catalog.
    """
    catalog = {
        "subjects": [
            {
                "id": s["id"],
                "dewey_id": s["dewey_id"],
                "name": s["name"],
                "description": s.get("description", ""),
                "state": s["state"],
            }
            for s in sorted(subjects, key=lambda x: x["dewey_id"])
        ],
        "last_updated": _now_iso(),
        "version": 1,
    }
    save_catalog(catalog, root)
    return catalog


# ---------------------------------------------------------------------------
# Anchor text rendering
# ---------------------------------------------------------------------------

def render_anchor_text(catalog: dict[str, Any], session_id: str | None = None) -> str:
    """
    Render the catalog as the text content of a session anchor message.
    Always begins with ANCHOR_MARKER on its own line.
    """
    lines: list[str] = [ANCHOR_MARKER]
    if session_id:
        lines.append(f"Session: {session_id}")
    lines.append(f"Last updated: {catalog.get('last_updated', 'unknown')}")
    lines.append("")
    subjects = catalog.get("subjects", [])
    if not subjects:
        lines.append("(no subjects registered yet)")
    else:
        lines.append("Subject catalog:")
        for s in subjects:
            state_tag = f"[{s['state']}]" if s.get("state") else ""
            desc = s.get("description", "")
            desc_part = f" - {desc}" if desc else ""
            lines.append(f"  {s['dewey_id']}  {s['name']} {state_tag}{desc_part}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session file anchor management
# ---------------------------------------------------------------------------

def _read_session_turns(session_path: Path) -> tuple[list[Any], bool]:
    """
    Read a session file and return (turns, is_dict_format).
    Handles both dict-with-history (current) and bare-list (legacy) formats.
    Returns ([], False) on any read/parse error.
    """
    if not session_path.exists():
        return [], False
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("history", []), True
        if isinstance(data, list):
            return data, False
        return [], False
    except (json.JSONDecodeError, OSError):
        return [], False


def _write_session_file(
    session_path: Path, turns: list[Any], original_data: Any = None
) -> None:
    """
    Write turns back to a session file, preserving the dict wrapper if the
    file was originally in dict-with-history format.
    """
    if isinstance(original_data, dict):
        original_data["history"] = turns
        out: Any = original_data
    else:
        out = turns
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _read_session_file(session_path: Path) -> list[Any]:
    """Legacy helper kept for backward compatibility. Returns turns only."""
    turns, _ = _read_session_turns(session_path)
    return turns


def _make_anchor_turn(anchor_text: str) -> dict[str, Any]:
    """Wrap anchor text as a Continue-compatible turn object."""
    return {
        "message": {
            "role": "user",
            "content": anchor_text,
            "id": "hubris-catalog-anchor",
        },
        "contextItems": [],
    }


def _content_to_str(content: Any) -> str:
    """Normalize turn content (string or list-of-parts) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return next(
            (item.get("text", "") for item in content
             if isinstance(item, dict) and item.get("type") == "text"),
            "",
        )
    return ""


def check_anchor(session_path: Path) -> bool:
    """
    Return True if the session file contains the catalog anchor marker.
    The anchor must appear in the content of the FIRST turn in the file.
    Handles both string and list-of-parts content.
    """
    turns, _ = _read_session_turns(session_path)
    if not turns:
        return False
    first = turns[0]
    if not isinstance(first, dict):
        return False
    msg = first.get("message", {})
    content = _content_to_str(msg.get("content", ""))
    stripped = content.lstrip()
    return stripped.startswith(ANCHOR_MARKER)


def inject_anchor(
    session_path: Path,
    catalog: dict[str, Any],
    replace_existing: bool = True,
) -> None:
    """
    Insert (or replace) the catalog anchor as the first turn in the session file.
    Preserves the dict-with-history wrapper if the file uses the current format.
    """
    if not session_path.exists():
        return
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        original_data = []

    turns, is_dict = (
        (original_data.get("history", []), True)
        if isinstance(original_data, dict)
        else (original_data if isinstance(original_data, list) else [], False)
    )

    anchor_text = render_anchor_text(catalog, session_id=session_path.stem)
    anchor_turn = _make_anchor_turn(anchor_text)

    if turns and isinstance(turns[0], dict):
        first_content = _content_to_str(turns[0].get("message", {}).get("content", ""))
        first_is_anchor = first_content.lstrip().startswith(ANCHOR_MARKER)
        if first_is_anchor and replace_existing:
            turns[0] = anchor_turn
        elif not first_is_anchor:
            turns.insert(0, anchor_turn)
        # else: anchor present and replace_existing=False - leave it
    else:
        turns.insert(0, anchor_turn)

    _write_session_file(session_path, turns, original_data)


def restore_anchor_after_truncation(
    session_path: Path,
    catalog: dict[str, Any],
    summary_text: str | None = None,
) -> None:
    """
    Restore the catalog anchor verbatim from the provided catalog dict,
    then optionally inject a compaction summary as the second turn.
    Preserves the dict-with-history wrapper if the file uses the current format.
    """
    if not session_path.exists():
        return
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        original_data = []

    turns: list = (
        original_data.get("history", [])
        if isinstance(original_data, dict)
        else (original_data if isinstance(original_data, list) else [])
    )

    anchor_text = render_anchor_text(catalog, session_id=session_path.stem)
    anchor_turn = _make_anchor_turn(anchor_text)

    turns.insert(0, anchor_turn)

    if summary_text:
        summary_turn = {
            "message": {
                "role": "user",
                "content": (
                    "[HUBRIS-TRUNCATION-RECOVERY]\n"
                    "The session was truncated. Summary of dropped content:\n\n"
                    + summary_text
                ),
                "id": "hubris-truncation-recovery",
            },
            "contextItems": [],
        }
        turns.insert(1, summary_turn)

    _write_session_file(session_path, turns, original_data)


# ---------------------------------------------------------------------------
# Workspace URI utilities
# ---------------------------------------------------------------------------

def decode_workspace_uri(uri: str) -> str:
    """
    Decode a Continue workspaceDirectory URI to a normalized filesystem path.

    Continue encodes the workspace as a percent-encoded file URI, e.g.
        file:///c%3A/Development/my-project
    This decodes to:
        c:/Development/my-project   (forward-slash form)
    The result is suitable for case-insensitive prefix matching against
    the paths stored in the watched_workspaces config list.
    """
    if uri.startswith("file:///"):
        return unquote(uri[len("file:///"):])
    return unquote(uri)


# ---------------------------------------------------------------------------
# Batch anchor injection
# ---------------------------------------------------------------------------

def inject_anchor_all_sessions(adapter: Any, cat: dict[str, Any]) -> None:
    """
    Inject or refresh the catalog anchor in all known session files.
    Best-effort: silently skips sessions with no writable path (e.g. GHCP
    read-only transcripts) and files that cannot be written.
    """
    for sid in adapter.list_sessions():
        path = adapter.get_session_path(sid)
        if path is None:
            continue
        try:
            inject_anchor(path, cat, replace_existing=True)
        except Exception:
            pass
