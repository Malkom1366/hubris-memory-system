"""
GitHub Copilot Chat front-end adapter for HuBrIS.

Reads JSONL transcript files from VS Code workspaceStorage and exposes them
through the SessionAdapter protocol defined in frontend_adapters.py.
"""

import json
import os
import sqlite3
from pathlib import Path

from backend_adapters import build_backend_adapter
from log import get_logger

_log = get_logger("hubris.adapter.ghcp")

_FENCE_START = "<!-- HUBRIS:START -->"
_FENCE_END = "<!-- HUBRIS:END -->"


def _write_hubris_fence(instructions_path: Path, content: str) -> None:
    """
    Write content into the HUBRIS fence block in instructions_path.
    Creates the file and fence markers if they don't exist.
    Writes atomically using a temp file and os.replace.
    """
    if instructions_path.exists():
        current = instructions_path.read_text(encoding="utf-8")
    else:
        instructions_path.parent.mkdir(parents=True, exist_ok=True)
        current = ""

    block = f"{_FENCE_START}\n{content.strip()}\n{_FENCE_END}"

    if _FENCE_START in current and _FENCE_END in current:
        start_idx = current.index(_FENCE_START)
        end_idx = current.index(_FENCE_END) + len(_FENCE_END)
        new_content = current[:start_idx] + block + current[end_idx:]
    elif current:
        new_content = block + "\n\n" + current.lstrip()
    else:
        new_content = block + "\n"

    tmp_path = instructions_path.with_suffix(".hubris_tmp")
    tmp_path.write_text(new_content, encoding="utf-8")
    os.replace(tmp_path, instructions_path)


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

    ADAPTER_NAME = "ghcp"

    def __init__(self, workspace_storage_dir: str | Path | None = None) -> None:
        if not workspace_storage_dir:
            appdata = os.environ.get("APPDATA") or str(
                Path.home() / "AppData" / "Roaming"
            )
            self._wsd = Path(appdata) / "Code" / "User" / "workspaceStorage"
        else:
            self._wsd = Path(workspace_storage_dir)
        self._paths: dict[str, Path] = {}
        self._title_cache: dict[str, str] | None = None

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

    def _build_title_cache(self) -> dict[str, str]:
        """
        Read chat.ChatSessionStore.index from every workspace state.vscdb
        and return a merged {session_id: title} dict.

        VS Code stores AI-generated session titles in the ItemTable of each
        workspace's state.vscdb under the key 'chat.ChatSessionStore.index'.
        The value is JSON: {"version": 1, "entries": {<uuid>: {"title": "...", ...}, ...}}.
        The same session can appear in multiple workspace databases (VS Code
        replicates the index); last writer wins on collisions since titles are
        identical across copies.
        """
        titles: dict[str, str] = {}
        if not self._wsd.is_dir():
            return titles
        for ws_dir in self._wsd.iterdir():
            db_path = ws_dir / "state.vscdb"
            if not db_path.is_file():
                continue
            try:
                con = sqlite3.connect(str(db_path))
                row = con.execute(
                    "SELECT value FROM ItemTable WHERE key='chat.ChatSessionStore.index'"
                ).fetchone()
                con.close()
                if not row:
                    continue
                data = json.loads(row[0])
                entries = data.get("entries", {}) if isinstance(data, dict) else {}
                for sid, meta in entries.items():
                    if isinstance(meta, dict):
                        t = meta.get("title")
                        if isinstance(t, str) and t.strip():
                            titles[sid] = t.strip()
            except Exception:
                pass
        return titles

    def get_session_title(self, session_id: str) -> str | None:
        """
        Return the VS Code AI-generated title for the session.

        Titles come from chat.ChatSessionStore.index in each workspace's
        state.vscdb. The cache is built once per adapter instance on first
        call and reused for all subsequent calls.
        Returns None if the session has no title entry.
        """
        if self._title_cache is None:
            self._title_cache = self._build_title_cache()
        return self._title_cache.get(session_id)

    def get_watch_dirs(self) -> list[Path]:
        """Return the VS Code workspaceStorage root as the single watched location."""
        return [self._wsd]

    def push_context(
        self,
        session_id: str,
        all_messages: list[dict],
        prev_count: int,
        cfg: dict,
        root: Path,
    ) -> None:
        """
        Distill recent session messages and write a HUBRIS fence block into the
        copilot-instructions.md file configured at cfg["ghcp_adapter"]["instructions_path"].

        No-op when:
        - ghcp_adapter.enabled is False (default)
        - instructions_path is not set
        - all_messages is empty
        - the backend adapter call fails or returns nothing
        """
        ghcp_cfg = cfg.get("ghcp_adapter", {})
        if not ghcp_cfg.get("enabled", False):
            return
        instructions_path_str = ghcp_cfg.get("instructions_path", "")
        if not instructions_path_str:
            return
        if not all_messages:
            return

        instructions_path = Path(instructions_path_str)
        token_limit = int(ghcp_cfg.get("injection_token_limit", 600))

        # Distill the most recent messages into a compact summary.
        recent = all_messages[-20:]
        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:500]}" for m in recent
        )

        system_prompt = (
            "You are HuBrIS, a memory system for AI coding assistants. "
            "Summarize the key context from this conversation that a coding assistant "
            "needs to know right now. Be concise and structured. Focus on: "
            "active tasks, recent decisions, important constraints, and what to do next. "
            f"Keep the total output under {token_limit} tokens."
        )
        user_prompt = (
            f"Conversation excerpt:\n{convo_text}\n\n"
            "Write a brief HuBrIS context block. Use markdown. Be dense and useful."
        )

        model = cfg.get("subagent_model") or cfg.get("meta_model", "")
        if not model:
            _log.warning(
                "GHCP PUSH_CONTEXT %s: no model configured, skipping", session_id[:8]
            )
            return

        try:
            result = build_backend_adapter(cfg).complete(system_prompt, user_prompt, model)
        except Exception as exc:
            _log.warning("GHCP PUSH_CONTEXT %s: backend error: %s", session_id[:8], exc)
            return

        if not result:
            return

        _write_hubris_fence(instructions_path, result)
        _log.info(
            "GHCP PUSH_CONTEXT %s: fence written to %s", session_id[:8], instructions_path
        )

    def workspace_id_allowed(self, workspace_id: str | None, cfg: dict) -> bool:
        """
        Return True if workspace_id passes the watched_workspace_hashes filter.

        Copilot workspace IDs are opaque workspaceStorage hashes. If no
        watched_workspace_hashes list is configured, all workspaces are allowed.
        """
        watched_hashes = cfg.get("watched_workspace_hashes", [])
        if not watched_hashes:
            return True
        return bool(workspace_id) and workspace_id in watched_hashes

    def install_rules(self, cfg: dict) -> None:
        """
        Copy hubris-guide.md into the .github/ directory of each workspace listed
        in copilot_workspace_roots. Called once at server startup so a Copilot
        session always has the guide without any manual file placement. If the
        destination already has identical content this is a no-op. Any stale
        version is replaced so the guide stays current with the codebase.
        """
        source = Path(__file__).parent / "hubris-guide.md"
        if not source.exists():
            _log.warning(
                "COPILOT RULES INSTALL: source file %s not found - skipping", source
            )
            return
        roots = cfg.get("copilot_workspace_roots", [])
        if not roots:
            return
        content = source.read_text(encoding="utf-8")
        for workspace_root in roots:
            github_dir = Path(workspace_root) / ".github"
            if not github_dir.exists():
                _log.warning(
                    "COPILOT RULES INSTALL: .github dir not found at %s - skipping",
                    github_dir,
                )
                continue
            dest = github_dir / "hubris-guide.md"
            if dest.exists() and dest.read_text(encoding="utf-8") == content:
                _log.debug("COPILOT RULES INSTALL: %s already up to date", dest)
                continue
            dest.write_text(content, encoding="utf-8")
            _log.info(
                "COPILOT RULES INSTALL: wrote %s (%d bytes)",
                dest,
                len(content.encode("utf-8")),
            )
