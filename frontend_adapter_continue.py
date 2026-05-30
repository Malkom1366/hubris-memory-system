"""
Continue AI front-end adapter for HuBrIS.

Reads session files from ~/.continue/sessions/ and exposes them through
the SessionAdapter protocol defined in frontend_adapters.py.
"""

import json
import os
from pathlib import Path

import catalog
import db as _db
import subjects as _subjects
from log import get_logger

_log = get_logger("hubris.adapter.continue")


class ContinueAdapter:
    """
    Reads Continue sessions from ~/.continue/sessions/.

    Continue stores one JSON file per session. The file is a JSON array of
    turn objects. Each turn object has a "message" key with "role" and "content".
    "sessions.json" is the index file and is excluded from the session list.
    """

    ADAPTER_NAME = "continue"

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
            # Strip any anchor messages injected by HuBrIS. Continue's in-memory
            # state removes them on the next write, so counting them would cause drift.
            messages = [
                m for m in messages
                if not m.get("content", "").lstrip().startswith(catalog.ANCHOR_MARKER)
            ]
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

    def get_session_title(self, session_id: str) -> str | None:
        """
        Return the session title from the sessions.json index, or None if
        the index is missing, unreadable, or does not contain this session.
        """
        index_path = self._dir / "sessions.json"
        if not index_path.exists():
            return None
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if not isinstance(entries, list):
                return None
            for entry in entries:
                if isinstance(entry, dict) and entry.get("sessionId") == session_id:
                    raw = entry.get("title")
                    if not isinstance(raw, str) or not raw.strip():
                        return None
                    text = raw.strip().replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
                    return text[:80] + ("..." if len(text) > 80 else "")
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

    def get_watch_dirs(self) -> list[Path]:
        """Return the Continue sessions directory as the single watched location."""
        return [self._dir]

    def push_context(
        self,
        session_id: str,
        all_messages: list[dict],
        prev_count: int,
        cfg: dict,
        root: Path,
    ) -> None:
        """
        Anchor management for Continue sessions.

        Handles anchor presence check, truncation recovery, first-sight
        injection, and anchor refresh after classification is enqueued.
        This is the Continue-specific behavior moved from server._on_session_changed.
        """
        session_path = self.get_session_path(session_id)
        if session_path is None:
            return

        cat = catalog.load_catalog(root)
        anchor_present = catalog.check_anchor(session_path)

        if not anchor_present and prev_count > 0:
            if len(all_messages) < prev_count:
                # Real truncation: Continue compacted the context, messages were lost.
                _log.warning(
                    "TRUNCATION DETECTED %s: prev_count=%d, current=%d - recovering",
                    session_id[:8], prev_count, len(all_messages),
                )
                dropped_count = prev_count - len(all_messages)
                existing_assignments = _db.load_assignments(session_id, root)
                dropped_subject_ids: set[str] = set()
                for msg_idx, subj_id in existing_assignments.items():
                    if msg_idx < dropped_count and subj_id is not None:
                        dropped_subject_ids.add(subj_id)
                if dropped_subject_ids:
                    all_subjects_snap = _subjects.load_subjects(root)
                    subject_lines = []
                    for sid in dropped_subject_ids:
                        s = next((x for x in all_subjects_snap if x["id"] == sid), None)
                        if s:
                            subject_lines.append(
                                f"  [{s['dewey_id']}] {s['name']} - use recall('{s['id']}')"
                            )
                    summary = (
                        f"HuBrIS truncation recovery: {dropped_count} message(s) were dropped by "
                        f"context window overflow. Content from these subjects is in long-term memory:\n"
                        + "\n".join(subject_lines)
                        + "\nAll subject memory is accessible via recall() or list_subjects."
                    )
                else:
                    summary = (
                        f"HuBrIS truncation recovery: {dropped_count} message(s) were dropped. "
                        f"No classified subject content was in the dropped range. "
                        f"Use list_subjects() to check available memory."
                    )
                catalog.restore_anchor_after_truncation(session_path, cat, summary_text=summary)
            else:
                # Anchor missing but no messages dropped - Continue overwrote the session
                # file. The refresh below will re-inject it.
                _log.debug(
                    "ANCHOR OVERWRITE %s: anchor removed by Continue (prev=%d, current=%d)"
                    " - will re-inject at refresh",
                    session_id[:8], prev_count, len(all_messages),
                )
        elif not anchor_present and prev_count == 0:
            # First time we have seen this session. Inject a fresh anchor.
            _log.info("ANCHOR INJECT (first-sight) session=%s", session_id[:8])
            catalog.inject_anchor(session_path, cat, replace_existing=False)

        # Refresh anchor so subject states stay current.
        rebuilt = catalog.rebuild_catalog_from_subjects(_subjects.load_subjects(root), root)
        if session_path.exists():
            _log.info(
                "ANCHOR REFRESH %s: %d subject(s) in ToC",
                session_id[:8], len(rebuilt.get("subjects", [])),
            )
            catalog.inject_anchor(session_path, rebuilt, replace_existing=True)

    def workspace_id_allowed(self, workspace_id: str | None, cfg: dict) -> bool:
        """
        Return True if workspace_id passes the watched_workspaces filter.

        Continue workspace IDs are percent-encoded file:// URIs. If no
        watched_workspaces list is configured, all workspaces are allowed.
        """
        watched = cfg.get("watched_workspaces", [])
        if not watched:
            return True
        if not workspace_id:
            return True
        decoded = catalog.decode_workspace_uri(workspace_id)
        return any(
            decoded.lower().startswith(w.lower().rstrip("\\/"))
            for w in watched
        )

    def install_rules(self, cfg: dict) -> None:  # noqa: ARG002 - cfg unused for Continue
        """
        Write hubris-guide.md from the project directory into ~/.continue/rules/.

        Called once at server startup so a fresh environment always has the guide
        without any manual file placement. The rules dir is created if absent.
        If the destination already has identical content this is a no-op. Any
        stale version is replaced so the guide stays current with the codebase.
        """
        source = Path(__file__).parent / "hubris-guide.md"
        if not source.exists():
            _log.warning("RULES INSTALL: source file %s not found - skipping", source)
            return
        rules_dir = Path.home() / ".continue" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        dest = rules_dir / "hubris-guide.md"
        content = source.read_text(encoding="utf-8")
        if dest.exists() and dest.read_text(encoding="utf-8") == content:
            _log.debug("RULES INSTALL: %s already up to date", dest)
            return
        dest.write_text(content, encoding="utf-8")
        _log.info("RULES INSTALL: wrote %s (%d bytes)", dest, len(content.encode("utf-8")))
