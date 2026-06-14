"""
KoboldCPP front-end adapter for HuBrIS.

Reads conversation save files produced by KoboldCPP's Kobold Lite chat UI
and exposes them through the SessionAdapter protocol defined in adapters.py.

Expected save directory structure:
    <saves_dir>/
        my-chat.json
        planescape-session-1.json
        ...

Session IDs are the filename stems (e.g. "planescape-session-1").

Two save formats are supported:

  Chat format (Kobold Lite instruct/chat mode) - primary:
    {
      "messages":   [{"role": "user"|"assistant"|"system", "content": "..."}],
      "systemprompt": "...",   <- synthesized into a system message if present
                                  and not already first in messages
      "chatname":   "..."
    }

  Story/actions format (KoboldAI legacy) - fallback:
    {
      "gamestate": {
        "prompt": "...",
        "actions": [
          {"text": "...", "actiontype": "story"},
          {"text": "...", "actiontype": ""}
        ]
      }
    }
    "story" actiontype = assistant output.
    "" or "prompt" actiontype = user input.
    "prompt" at gamestate root = synthesized as a system message (if non-empty).

Tombstone writing is supported for the chat format. Story format saves are
treated as read-only because reconstructing the actions array layout
accurately after a tombstone is fragile.

Configure in config.json:
  "active_adapters": ["koboldcpp"],
  "adapters": {
    "koboldcpp": {
      "saves_dir": "C:\\\\path\\\\to\\\\kobold\\\\saves"
    }
  }
"""

import json
import os
from pathlib import Path

import catalog
from log import get_logger

_log = get_logger("hubris.adapter.koboldcpp")

# KoboldAI story-mode action type that marks AI-generated content.
_STORY_AI_ACTIONTYPE = "story"


class KoboldCppAdapter:
    """
    Reads KoboldCPP conversation saves from a user-configured directory.

    KoboldCPP/Kobold Lite does not write to a fixed location; the user
    chooses where to save via the UI. Point saves_dir at that folder.

    Unlike Continue, saves are point-in-time snapshots written manually
    by the user. The watcher picks up a change each time the file's mtime
    updates (i.e. each manual save). Mid-session messages are NOT visible
    until the user saves.
    """

    ADAPTER_NAME = "koboldcpp"

    def __init__(self, saves_dir: str | Path | None = None) -> None:
        if not saves_dir:
            _log.warning(
                "KoboldCppAdapter: saves_dir not configured - falling back to cwd. "
                "Set adapters.koboldcpp.saves_dir in config.json."
            )
            self._dir = Path.cwd()
        else:
            self._dir = Path(saves_dir)

    # ------------------------------------------------------------------
    # SessionAdapter protocol
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[str]:
        """Return stems of all .json save files in the saves directory."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.json")]

    def read_messages(self, session_id: str) -> list[dict]:
        """
        Parse a KoboldCPP save file and return a flat list of
        {"role": str, "content": str} dicts, in order.

        Tries chat format first; falls back to story/actions format.
        Strips anchor messages injected by HuBrIS (same as ContinueAdapter).
        Returns [] on any parse error.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(data, dict):
            return []

        messages = self._parse_chat_format(data)
        if messages is None:
            messages = self._parse_story_format(data)
        if messages is None:
            _log.warning(
                "KoboldCppAdapter: unrecognised save format in %s.json", session_id
            )
            return []

        # Strip anchor messages injected by HuBrIS.
        messages = [
            m for m in messages
            if not m.get("content", "").lstrip().startswith(catalog.ANCHOR_MARKER)
        ]
        return messages

    def get_mtime(self, session_id: str) -> float:
        """Return the mtime of the save file, or 0.0 if absent."""
        path = self._dir / f"{session_id}.json"
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def list_historical_sessions(self) -> list[str]:
        """Return stems of all .json save files (same as list_sessions)."""
        return self.list_sessions()

    def get_session_path(self, session_id: str) -> Path | None:
        """Return the path to the save file for chat-format saves (write-capable)."""
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None
        # Only expose a writable path for chat format; story format is read-only.
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "messages" in data:
                return path
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def get_session_workspace(self, session_id: str) -> str | None:
        """KoboldCPP has no workspace concept. Always returns None."""
        return None

    def write_tombstones(
        self, session_id: str, tombstone_map: dict[int, str]
    ) -> None:
        """
        Replace the content of specific messages (by HuBrIS message_index) with
        tombstone text. Only operates on chat-format saves; story format is a no-op.

        message_index is 0-based after excluding system and anchor messages,
        matching the indexing scheme used by ContinueAdapter.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, dict) or "messages" not in data:
            return  # story format or unrecognised - no-op

        raw_messages: list = data.get("messages", [])
        if not isinstance(raw_messages, list):
            return

        # Build filtered_index -> raw_list_position mapping.
        # Mirrors the filtering in read_messages / _parse_chat_format:
        #   - non-dict entries are skipped
        #   - entries without role/content are skipped
        #   - anchor messages are skipped
        filtered_to_pos: dict[int, int] = {}
        filtered_idx = 0
        for pos, entry in enumerate(raw_messages):
            if not isinstance(entry, dict):
                continue
            role = entry.get("role", "")
            content = entry.get("content", "")
            if not role or content is None:
                continue
            content_str = content if isinstance(content, str) else json.dumps(content)
            if content_str.lstrip().startswith(catalog.ANCHOR_MARKER):
                continue
            filtered_to_pos[filtered_idx] = pos
            filtered_idx += 1

        modified = False
        for filt_idx, tombstone_text in tombstone_map.items():
            pos = filtered_to_pos.get(filt_idx)
            if pos is None:
                continue
            raw_messages[pos]["content"] = tombstone_text
            modified = True

        if not modified:
            return

        data["messages"] = raw_messages
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            _log.warning(
                "KoboldCppAdapter: failed to write tombstones to %s.json: %s",
                session_id, exc
            )

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    def _parse_chat_format(self, data: dict) -> list[dict] | None:
        """
        Try to parse a Kobold Lite chat-format save.

        Expected structure:
          {
            "messages": [{"role": "...", "content": "..."}, ...],
            "systemprompt": "..."   <- optional, synthesized if absent from messages
          }

        Returns a list of {"role", "content"} dicts, or None if the data does
        not match this format.
        """
        raw = data.get("messages")
        if not isinstance(raw, list):
            return None

        messages: list[dict] = []

        # If "systemprompt" exists at the root and messages do not already
        # start with a system role entry, prepend a synthetic system message.
        systemprompt = data.get("systemprompt", "")
        if (
            isinstance(systemprompt, str)
            and systemprompt.strip()
            and not (raw and isinstance(raw[0], dict) and raw[0].get("role") == "system")
        ):
            messages.append({"role": "system", "content": systemprompt.strip()})

        for entry in raw:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role", "")
            content = entry.get("content", "")
            if not role or content is None:
                continue
            if not isinstance(content, str):
                content = json.dumps(content)
            messages.append({"role": role, "content": content})

        return messages

    def _parse_story_format(self, data: dict) -> list[dict] | None:
        """
        Try to parse a KoboldAI story/actions-format save (legacy).

        Expected structure:
          {
            "gamestate": {
              "prompt": "...",
              "actions": [
                {"text": "...", "actiontype": "story"},
                ...
              ]
            }
          }

        "story" actiontype = assistant output.
        "" or "prompt" actiontype = user input.
        "prompt" at root of gamestate = synthesized as a system message if non-empty.

        Returns a list of {"role", "content"} dicts, or None if the data does
        not match this format.
        """
        gamestate = data.get("gamestate")
        if not isinstance(gamestate, dict):
            return None

        actions = gamestate.get("actions")
        if not isinstance(actions, list):
            return None

        messages: list[dict] = []

        intro = gamestate.get("prompt", "")
        if isinstance(intro, str) and intro.strip():
            messages.append({"role": "system", "content": intro.strip()})

        for action in actions:
            if not isinstance(action, dict):
                continue
            text = action.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            actiontype = action.get("actiontype", "")
            if actiontype == _STORY_AI_ACTIONTYPE:
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": text.strip()})

        return messages if messages else None
