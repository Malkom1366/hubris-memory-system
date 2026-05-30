"""
Front-end session adapters for HuBrIS.

This module is the single source of truth for all front-end adapters - the
components that read AI chat session files and expose them as a uniform list
of messages to the rest of HuBrIS.

Contents:
  SessionAdapter   - Protocol / interface all front-end adapters must satisfy.
  ContinueAdapter  - Reads ~/.continue/sessions/*.json (Continue AI).
  VSCodeCopilotAdapter - Reads VS Code workspaceStorage JSONL transcripts
                         (GitHub Copilot Chat).
  build_frontend_adapter(cfg) - Factory: reads 'adapter' from a config dict
                                and returns the correct concrete adapter.
                                This is the preferred instantiation path;
                                server.py and watcher.py should call this
                                rather than constructing adapters directly.

Continue session files may be in one of two formats:
  - Legacy (bare list): a JSON array of turn objects at the root.
  - Current (dict with history key): a JSON object with a "history" key whose
    value is the array of turn objects. Other top-level keys like "sessionId",
    "title", and "workspaceDirectory" are present but ignored by HuBrIS.

Each turn has a "message" key with "role" and "content" fields. In newer
versions, user turn content is a list of {"type": "text", "text": "..."}
parts rather than a plain string. HuBrIS flattens these to a single string.
"""

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionAdapter(Protocol):
    """Minimum interface any session source must satisfy."""

    def list_sessions(self) -> list[str]:
        """Return a list of session IDs (stable identifiers, suitable as dict keys)."""
        ...

    def read_messages(self, session_id: str) -> list[dict]:
        """
        Return the current messages for the session as a list of dicts.
        Each dict has at least "role" (str) and "content" (str).
        Returns an empty list if the session cannot be read.
        """
        ...

    def get_mtime(self, session_id: str) -> float:
        """Return the last-modified timestamp (epoch float) for the session file."""
        ...

    def list_historical_sessions(self) -> list[str]:
        """
        Return a list of ALL known session IDs, including sessions that are
        not currently active. Used during first-run to identify pre-existing
        sessions that should be blacklisted automatically.
        """
        ...

    def get_session_path(self, session_id: str) -> Path | None:
        """
        Return the filesystem path for the session file, or None if the
        adapter's sessions are read-only (e.g. VS Code Copilot transcripts).
        Callers must gate any write operations (anchor inject, etc.) on a
        non-None return value.
        """
        ...

    def get_session_workspace(self, session_id: str) -> str | None:
        """
        Return an opaque workspace identifier for the given session, or None
        if the adapter has no workspace concept (e.g. Continue sessions are
        global). For VSCodeCopilotAdapter this is the workspaceStorage hash.
        """
        ...

    def write_tombstones(
        self, session_id: str, tombstone_map: dict[int, str]
    ) -> None:
        """
        Replace the content of specific messages (by their HuBrIS message_index)
        with tombstone text. The turn count in the session file is preserved;
        only the content of targeted messages changes.

        tombstone_map maps HuBrIS message_index (0-based, anchor-excluded) to
        the replacement text string.

        Adapters that cannot write to the session file (e.g. VSCodeCopilotAdapter)
        must implement this as a no-op.
        """
        ...


class ContinueAdapter:
    """
    Reads Continue sessions from ~/.continue/sessions/.

    Continue stores one JSON file per session. The file is a JSON array of
    turn objects. Each turn object has a "message" key with "role" and "content".
    "sessions.json" is the index file and is excluded from the session list.
    """

    def __init__(self, sessions_dir: str | Path | None = None) -> None:
        if sessions_dir is None:
            self._dir = Path.home() / ".continue" / "sessions"
        else:
            self._dir = Path(sessions_dir)

    def list_sessions(self) -> list[str]:
        """
        Return session IDs for all .json files in the sessions directory,
        excluding sessions.json (the index).
        """
        if not self._dir.exists():
            return []
        return [
            p.stem
            for p in self._dir.glob("*.json")
            if p.name != "sessions.json"
        ]

    def read_messages(self, session_id: str) -> list[dict]:
        """
        Parse a Continue session file and return a flat list of
        {"role": str, "content": str} dicts, in order.
        Handles both the legacy bare-list format and the current
        dict-with-history format. Flattens list-style content parts.
        Returns [] on any parse error.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support both dict-with-history (current) and bare-list (legacy).
            if isinstance(data, dict):
                turns = data.get("history", [])
            elif isinstance(data, list):
                turns = data
            else:
                return []
            messages: list[dict] = []
            for turn in turns:
                msg = turn.get("message") if isinstance(turn, dict) else None
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not role or content is None:
                    continue
                # Newer user turns store content as [{"type": "text", "text": "..."}].
                if isinstance(content, list):
                    parts = [
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                        if not isinstance(item, dict) or item.get("type") == "text"
                    ]
                    content = "\n".join(parts)
                elif not isinstance(content, str):
                    content = json.dumps(content)
                messages.append({"role": role, "content": content})
            return messages
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return []

    def get_mtime(self, session_id: str) -> float:
        """Return the mtime of the session file, or 0.0 if absent."""
        path = self._dir / f"{session_id}.json"
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def list_historical_sessions(self) -> list[str]:
        """Return stems of all .json files in the sessions directory, excluding sessions.json."""
        if not self._dir.exists():
            return []
        return [
            p.stem
            for p in self._dir.glob("*.json")
            if p.name != "sessions.json"
        ]

    def get_session_path(self, session_id: str) -> Path | None:
        """Return the path to the session .json file."""
        return self._dir / f"{session_id}.json"

    def get_session_workspace(self, session_id: str) -> str | None:
        """Return the workspaceDirectory field from the Continue session file, or None."""
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get("workspaceDirectory") or None
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def write_tombstones(
        self, session_id: str, tombstone_map: dict[int, str]
    ) -> None:
        """
        Replace the content of messages at the given HuBrIS message indices
        with tombstone text, in-place in the session JSON file.

        Anchor turns (content starts with "[HUBRIS-CATALOG-ANCHOR]") are not
        counted as HuBrIS message indices - the mapping skips them so that
        index 0 refers to the first non-anchor message in the file.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if isinstance(original_data, dict):
            raw_turns: list = original_data.get("history", [])
        elif isinstance(original_data, list):
            raw_turns = original_data
        else:
            return

        # Build mapping: filtered_index (HuBrIS message index) -> turn list position.
        # The mapping mirrors the filtering done in _on_session_changed:
        #   - turns missing a valid message dict are skipped
        #   - turns whose content starts with the anchor marker are skipped
        _ANCHOR_MARKER = "[HUBRIS-CATALOG-ANCHOR]"
        filtered_to_pos: dict[int, int] = {}
        filtered_idx = 0
        for pos, turn in enumerate(raw_turns):
            msg = turn.get("message") if isinstance(turn, dict) else None
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not role or content is None:
                continue
            # Normalise content to string for anchor check.
            if isinstance(content, list):
                content_str = "\n".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            else:
                content_str = str(content)
            if content_str.lstrip().startswith(_ANCHOR_MARKER):
                continue  # anchor - not a HuBrIS message index
            filtered_to_pos[filtered_idx] = pos
            filtered_idx += 1

        modified = False
        for filt_idx, tombstone_text in tombstone_map.items():
            pos = filtered_to_pos.get(filt_idx)
            if pos is None:
                continue
            raw_turns[pos]["message"]["content"] = tombstone_text
            modified = True

        if not modified:
            return

        if isinstance(original_data, dict):
            original_data["history"] = raw_turns
            out: object = original_data
        else:
            out = raw_turns
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


class VSCodeCopilotAdapter:
    """
    Reads GitHub Copilot Chat sessions from VS Code's workspaceStorage.

    VS Code writes one JSONL file per session at:
        %APPDATA%\\Code\\User\\workspaceStorage\\<workspace-hash>\\
            GitHub.copilot-chat\\transcripts\\<session-id>.jsonl

    Each line is a JSON event object. Relevant event types:
        - user.message:      data.content (str) - the user's message text
        - assistant.message: data.content (str) - skipped when empty (tool-only turns)

    Session IDs are the bare UUID stems of the .jsonl filenames. UUID4 values
    are globally unique so no workspace-hash prefix is needed.

    An internal path cache (_paths) is refreshed on each list_sessions() call.
    read_messages() and get_mtime() fall back to a fresh scan if the session
    is not yet in the cache (handles calls before the first list_sessions()).
    """

    def __init__(self, workspace_storage_dir: str | Path | None = None) -> None:
        if workspace_storage_dir is None:
            appdata = os.environ.get("APPDATA") or str(
                Path.home() / "AppData" / "Roaming"
            )
            self._wsd = Path(appdata) / "Code" / "User" / "workspaceStorage"
        else:
            self._wsd = Path(workspace_storage_dir)
        self._paths: dict[str, Path] = {}

    def _scan(self) -> None:
        """Refresh _paths by scanning all workspace transcript directories."""
        self._paths = {}
        if not self._wsd.is_dir():
            return
        for ws_dir in self._wsd.iterdir():
            if not ws_dir.is_dir():
                continue
            transcripts_dir = ws_dir / "GitHub.copilot-chat" / "transcripts"
            if not transcripts_dir.is_dir():
                continue
            for jsonl in transcripts_dir.glob("*.jsonl"):
                # Last writer wins on UUID collision (astronomically unlikely).
                self._paths[jsonl.stem] = jsonl

    def list_sessions(self) -> list[str]:
        """
        Discover all JSONL transcript files under workspaceStorage and return
        their UUID stems as session IDs. Refreshes the internal path cache.
        """
        self._scan()
        return list(self._paths.keys())

    def read_messages(self, session_id: str) -> list[dict]:
        """
        Parse a GHCP JSONL transcript and return a flat list of
        {"role": str, "content": str} dicts in chronological order.

        Only user.message and assistant.message events with non-empty content
        are included. Tool-only assistant turns (empty content) are skipped.
        Returns [] on any parse error or if the session cannot be found.
        """
        if session_id not in self._paths:
            self._scan()
        path = self._paths.get(session_id)
        if path is None or not path.exists():
            return []
        messages: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")
                    data = event.get("data") or {}
                    if event_type == "user.message":
                        content = data.get("content", "")
                        if content:
                            messages.append({"role": "user", "content": content})
                    elif event_type == "assistant.message":
                        content = data.get("content", "")
                        if content:  # skip tool-only turns where content is empty
                            messages.append({"role": "assistant", "content": content})
        except OSError:
            return []
        return messages

    def get_mtime(self, session_id: str) -> float:
        """Return the mtime of the session file, or 0.0 if not in cache."""
        if session_id not in self._paths:
            self._scan()
        path = self._paths.get(session_id)
        if path is None:
            return 0.0
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def list_historical_sessions(self) -> list[str]:
        """Scan all workspaceStorage transcripts and return their UUID stems."""
        self._scan()
        return list(self._paths.keys())

    def get_session_path(self, session_id: str) -> None:  # type: ignore[override]
        """
        GHCP transcripts are written by VS Code and are read-only.
        Always returns None to signal callers that write operations
        (anchor injection, etc.) are not supported.
        """
        return None

    def write_tombstones(
        self, session_id: str, tombstone_map: dict[int, str]
    ) -> None:
        """
        GHCP transcripts are read-only. This is a no-op.
        Subject memory is still crystallised by the handler; the actual
        session file cannot be rewritten until a write path is established.
        """
        return

    def get_session_workspace(self, session_id: str) -> str | None:
        """
        Return the workspaceStorage hash for the given session, or None if
        the session is not found after a fresh scan. The hash is the name
        of the intermediate directory:
            workspaceStorage/<hash>/GitHub.copilot-chat/transcripts/<id>.jsonl
        """
        path = self._paths.get(session_id)
        if path is None:
            self._scan()
            path = self._paths.get(session_id)
        if path is None:
            return None
        # parents[0] = transcripts/, parents[1] = GitHub.copilot-chat/, parents[2] = <hash>/
        return path.parents[2].name


def build_frontend_adapter(cfg: dict) -> SessionAdapter:
    """
    Return the configured front-end session adapter.

    Reads 'adapter' from cfg. Recognised values:
      'continue' - ContinueAdapter reading ~/.continue/sessions/ (or
                   cfg['sessions_dir'] if set).
      'copilot'  - VSCodeCopilotAdapter reading VS Code workspaceStorage
                   JSONL transcripts (cfg['workspace_storage_dir'] if set).

    Raises ValueError for any unrecognised adapter name.
    """
    name = cfg.get("adapter", "continue")
    if name == "copilot":
        wsd = cfg.get("workspace_storage_dir") or None
        return VSCodeCopilotAdapter(workspace_storage_dir=wsd)
    if name == "continue":
        return ContinueAdapter(sessions_dir=cfg.get("sessions_dir"))
    raise ValueError(
        f"Unknown frontend adapter: {name!r}. Expected 'continue' or 'copilot'."
    )
