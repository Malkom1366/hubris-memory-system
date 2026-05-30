"""
Tests for frontend_adapters.py - ContinueAdapter, VSCodeCopilotAdapter,
build_frontend_adapter, build_active_frontend_adapters, and registry.py.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from frontend_adapters import (
    ContinueAdapter,
    VSCodeCopilotAdapter,
    build_frontend_adapter,
    build_active_frontend_adapters,
)


def _write_session(dir: Path, name: str, turns: list[dict]) -> Path:
    path = dir / name
    path.write_text(json.dumps(turns), encoding="utf-8")
    return path


def _make_turn(role: str, content: str, extra: dict | None = None) -> dict:
    turn = {
        "message": {"role": role, "content": content, "id": "abc"},
        "contextItems": [],
    }
    if extra:
        turn.update(extra)
    return turn


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_returns_stem_names(self, tmp_path):
        _write_session(tmp_path, "abc123.json", [])
        _write_session(tmp_path, "def456.json", [])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        result = adapter.list_sessions()
        assert set(result) == {"abc123", "def456"}

    def test_excludes_sessions_json(self, tmp_path):
        _write_session(tmp_path, "sessions.json", [])
        _write_session(tmp_path, "real.json", [])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        result = adapter.list_sessions()
        assert "sessions" not in result
        assert "real" in result

    def test_empty_dir(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.list_sessions() == []

    def test_missing_dir(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path / "nonexistent"))
        assert adapter.list_sessions() == []


# ---------------------------------------------------------------------------
# read_messages
# ---------------------------------------------------------------------------

class TestReadMessages:
    def test_parses_role_and_content(self, tmp_path):
        turns = [
            _make_turn("user", "Hello"),
            _make_turn("assistant", "Hi there"),
        ]
        _write_session(tmp_path, "s1.json", turns)
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        msgs = adapter.read_messages("s1")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there"}

    def test_handles_non_string_content(self, tmp_path):
        turns = [
            {"message": {"role": "user", "content": [{"type": "text", "text": "hello"}], "id": "x"}, "contextItems": []},
        ]
        _write_session(tmp_path, "s2.json", turns)
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        msgs = adapter.read_messages("s2")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # content should be the json-dumped list, not crash
        assert isinstance(msgs[0]["content"], str)

    def test_skips_turns_missing_message(self, tmp_path):
        turns = [
            _make_turn("user", "Valid"),
            {"contextItems": []},  # no "message" key
            _make_turn("assistant", "Also valid"),
        ]
        _write_session(tmp_path, "s3.json", turns)
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        msgs = adapter.read_messages("s3")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Valid"
        assert msgs[1]["content"] == "Also valid"

    def test_missing_file_returns_empty(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.read_messages("nonexistent") == []

    def test_empty_array_returns_empty(self, tmp_path):
        _write_session(tmp_path, "empty.json", [])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.read_messages("empty") == []

    def test_strips_anchor_messages(self, tmp_path):
        """Anchor messages injected by HuBrIS must be excluded from the returned list."""
        import catalog
        anchor_content = catalog.ANCHOR_MARKER + " some catalog content here"
        turns = [
            _make_turn("user", "Real message"),
            _make_turn("assistant", anchor_content),
            _make_turn("user", "Another real message"),
        ]
        _write_session(tmp_path, "anchored.json", turns)
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        msgs = adapter.read_messages("anchored")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Real message"
        assert msgs[1]["content"] == "Another real message"


# ---------------------------------------------------------------------------
# get_mtime
# ---------------------------------------------------------------------------

class TestGetMtime:
    def test_returns_float(self, tmp_path):
        path = _write_session(tmp_path, "t.json", [])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        mtime = adapter.get_mtime("t")
        assert isinstance(mtime, float)
        assert mtime == os.path.getmtime(str(path))

    def test_missing_file_returns_zero(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_mtime("missing") == 0.0


# ===========================================================================
# VSCodeCopilotAdapter helpers
# ===========================================================================

def _make_ws_transcripts(base: Path, ws_hash: str) -> Path:
    """Create a fake workspaceStorage/<ws_hash>/GitHub.copilot-chat/transcripts/ dir."""
    transcripts = base / ws_hash / "GitHub.copilot-chat" / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    return transcripts


def _write_jsonl(transcripts_dir: Path, session_id: str, events: list[dict]) -> Path:
    path = transcripts_dir / f"{session_id}.jsonl"
    lines = [json.dumps(e) for e in events]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _user_event(content: str) -> dict:
    return {"type": "user.message", "data": {"content": content, "attachments": []}, "id": "u1", "timestamp": "2026-01-01T00:00:00Z", "parentId": None}


def _assistant_event(content: str, has_tool_requests: bool = False) -> dict:
    data: dict = {"messageId": "a1", "content": content, "reasoningText": ""}
    if has_tool_requests:
        data["toolRequests"] = [{"toolCallId": "tc1", "name": "some_tool", "arguments": "{}", "type": "function"}]
    return {"type": "assistant.message", "data": data, "id": "a1", "timestamp": "2026-01-01T00:00:01Z", "parentId": "u1"}


def _session_start_event(session_id: str) -> dict:
    return {"type": "session.start", "data": {"sessionId": session_id, "version": 1, "producer": "copilot-agent"}, "id": "s1", "timestamp": "2026-01-01T00:00:00Z", "parentId": None}


def _tool_start_event() -> dict:
    return {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "read_file", "arguments": {}}, "id": "t1", "timestamp": "2026-01-01T00:00:01Z", "parentId": "a1"}


# ===========================================================================
# VSCodeCopilotAdapter tests
# ===========================================================================

class TestVSCodeCopilotAdapterListSessions:
    def test_finds_sessions_across_workspaces(self, tmp_path):
        td1 = _make_ws_transcripts(tmp_path, "ws1abc")
        td2 = _make_ws_transcripts(tmp_path, "ws2def")
        _write_jsonl(td1, "session-aaa", [])
        _write_jsonl(td2, "session-bbb", [])
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        result = adapter.list_sessions()
        assert set(result) == {"session-aaa", "session-bbb"}

    def test_empty_workspace_storage(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.list_sessions() == []

    def test_missing_workspace_storage_dir(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path / "nonexistent"))
        assert adapter.list_sessions() == []

    def test_ignores_dirs_without_transcripts(self, tmp_path):
        (tmp_path / "ws1" / "other-extension").mkdir(parents=True)
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.list_sessions() == []

    def test_only_includes_jsonl_files(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        _write_jsonl(td, "session-uuid", [])
        (td / "notes.txt").write_text("ignored")
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        result = adapter.list_sessions()
        assert result == ["session-uuid"]


class TestVSCodeCopilotAdapterReadMessages:
    def test_parses_user_and_assistant_turns(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        events = [
            _session_start_event("sid1"),
            _user_event("Hello there"),
            _assistant_event("Hi back"),
        ]
        _write_jsonl(td, "sid1", events)
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        msgs = adapter.read_messages("sid1")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello there"}
        assert msgs[1] == {"role": "assistant", "content": "Hi back"}

    def test_skips_tool_only_assistant_turns(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        events = [
            _user_event("Do something"),
            _assistant_event("", has_tool_requests=True),  # tool-only, empty content
            _assistant_event("Done"),
        ]
        _write_jsonl(td, "sid2", events)
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        msgs = adapter.read_messages("sid2")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1] == {"role": "assistant", "content": "Done"}

    def test_skips_non_message_events(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        events = [
            _session_start_event("sid3"),
            _tool_start_event(),
            _user_event("Real message"),
        ]
        _write_jsonl(td, "sid3", events)
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        msgs = adapter.read_messages("sid3")
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "Real message"}

    def test_skips_empty_user_content(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        events = [_user_event(""), _user_event("Real")]
        _write_jsonl(td, "sid4", events)
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        msgs = adapter.read_messages("sid4")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Real"

    def test_tolerates_malformed_jsonl_lines(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        path = td / "sid5.jsonl"
        path.write_text(
            json.dumps(_user_event("Good")) + "\n"
            "NOT JSON AT ALL\n"
            + json.dumps(_assistant_event("Also good")) + "\n",
            encoding="utf-8",
        )
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        msgs = adapter.read_messages("sid5")
        assert len(msgs) == 2

    def test_missing_session_returns_empty(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.read_messages("does-not-exist") == []

    def test_auto_scans_if_not_in_cache(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        _write_jsonl(td, "sid6", [_user_event("Hi")])
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        # Call read_messages directly without calling list_sessions first.
        msgs = adapter.read_messages("sid6")
        assert len(msgs) == 1


class TestVSCodeCopilotAdapterGetMtime:
    def test_returns_file_mtime(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        path = _write_jsonl(td, "sid7", [])
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        adapter.list_sessions()
        mtime = adapter.get_mtime("sid7")
        assert isinstance(mtime, float)
        assert mtime == os.path.getmtime(str(path))

    def test_missing_session_returns_zero(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_mtime("nonexistent") == 0.0

    def test_auto_scans_if_not_in_cache(self, tmp_path):
        td = _make_ws_transcripts(tmp_path, "ws1")
        _write_jsonl(td, "sid8", [])
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        # Call get_mtime without list_sessions first.
        mtime = adapter.get_mtime("sid8")
        assert isinstance(mtime, float)
        assert mtime > 0.0


# ---------------------------------------------------------------------------
# build_frontend_adapter (shim - returns first active adapter)
# ---------------------------------------------------------------------------

class TestBuildFrontendAdapter:
    def test_continue_returns_continue_adapter(self, tmp_path):
        cfg = {
            "active_adapters": ["continue"],
            "adapters": {"continue": {"sessions_dir": str(tmp_path)}},
        }
        adapter = build_frontend_adapter(cfg)
        assert isinstance(adapter, ContinueAdapter)

    def test_ghcp_returns_vscode_adapter(self, tmp_path):
        cfg = {
            "active_adapters": ["ghcp"],
            "adapters": {"ghcp": {"workspace_storage_dir": str(tmp_path)}},
        }
        adapter = build_frontend_adapter(cfg)
        assert isinstance(adapter, VSCodeCopilotAdapter)

    def test_default_active_adapters_is_continue(self, tmp_path):
        cfg = {"adapters": {"continue": {"sessions_dir": str(tmp_path)}}}
        adapter = build_frontend_adapter(cfg)
        assert isinstance(adapter, ContinueAdapter)

    def test_unknown_adapter_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown frontend adapter"):
            build_frontend_adapter({"active_adapters": ["cursor"]})


# ---------------------------------------------------------------------------
# build_active_frontend_adapters
# ---------------------------------------------------------------------------

class TestBuildActiveFrontendAdapters:
    def test_returns_list_of_one(self, tmp_path):
        cfg = {
            "active_adapters": ["continue"],
            "adapters": {"continue": {"sessions_dir": str(tmp_path)}},
        }
        adapters = build_active_frontend_adapters(cfg)
        assert len(adapters) == 1
        assert isinstance(adapters[0], ContinueAdapter)

    def test_returns_multiple(self, tmp_path):
        cfg = {
            "active_adapters": ["continue", "ghcp"],
            "adapters": {
                "continue": {"sessions_dir": str(tmp_path)},
                "ghcp": {"workspace_storage_dir": str(tmp_path)},
            },
        }
        adapters = build_active_frontend_adapters(cfg)
        assert len(adapters) == 2
        assert isinstance(adapters[0], ContinueAdapter)
        assert isinstance(adapters[1], VSCodeCopilotAdapter)

    def test_empty_active_adapters_returns_empty_list(self):
        cfg = {"active_adapters": []}
        adapters = build_active_frontend_adapters(cfg)
        assert adapters == []


# ---------------------------------------------------------------------------
# registry - discover_frontend_adapters
# ---------------------------------------------------------------------------

class TestDiscoverFrontendAdapters:
    def test_discovers_built_in_adapters(self):
        from registry import discover_frontend_adapters
        discovered = discover_frontend_adapters()
        assert "continue" in discovered
        assert "ghcp" in discovered
        assert discovered["continue"] is ContinueAdapter
        assert discovered["ghcp"] is VSCodeCopilotAdapter

    def test_uses_adapter_name_attribute(self, tmp_path):
        """A synthetic module with ADAPTER_NAME should be discovered under that name."""
        from registry import discover_frontend_adapters
        mod_file = tmp_path / "frontend_adapter_synthetic.py"
        mod_file.write_text(
            "class SyntheticAdapter:\n"
            '    ADAPTER_NAME = "synthetic"\n'
            "    def __init__(self): pass\n",
            encoding="utf-8",
        )
        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            discovered = discover_frontend_adapters(search_dir=tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("frontend_adapter_synthetic", None)
        assert "synthetic" in discovered

    def test_falls_back_to_filename_suffix(self, tmp_path):
        """A class without ADAPTER_NAME is keyed by the filename suffix."""
        from registry import discover_frontend_adapters
        mod_file = tmp_path / "frontend_adapter_nametag.py"
        mod_file.write_text(
            "class NoNameAdapter:\n"
            "    def __init__(self): pass\n",
            encoding="utf-8",
        )
        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            discovered = discover_frontend_adapters(search_dir=tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("frontend_adapter_nametag", None)
        assert "nametag" in discovered

    def test_explicit_module_class_override(self, tmp_path):
        """Explicit module/class config keys override discovery."""
        from registry import build_active_frontend_adapters
        # Re-use ContinueAdapter but load it via explicit keys.
        cfg = {
            "active_adapters": ["custom"],
            "adapters": {
                "custom": {
                    "module": "frontend_adapter_continue",
                    "class": "ContinueAdapter",
                    "sessions_dir": str(tmp_path),
                },
            },
        }
        adapters = build_active_frontend_adapters(cfg)
        assert len(adapters) == 1
        assert isinstance(adapters[0], ContinueAdapter)

    def test_unknown_name_raises_value_error(self):
        from registry import build_active_frontend_adapters
        cfg = {"active_adapters": ["does_not_exist"]}
        with pytest.raises(ValueError, match="Unknown frontend adapter"):
            build_active_frontend_adapters(cfg)


# ---------------------------------------------------------------------------
# ContinueAdapter.get_session_title
# ---------------------------------------------------------------------------

class TestContinueAdapterGetSessionTitle:
    def _write_index(self, sessions_dir: Path, entries: list[dict]) -> None:
        (sessions_dir / "sessions.json").write_text(
            json.dumps(entries), encoding="utf-8"
        )

    def test_returns_title_for_known_session(self, tmp_path):
        self._write_index(tmp_path, [
            {"sessionId": "abc", "title": "My Great Session", "messageCount": 3},
        ])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") == "My Great Session"

    def test_returns_none_for_unknown_session(self, tmp_path):
        self._write_index(tmp_path, [
            {"sessionId": "abc", "title": "Something", "messageCount": 1},
        ])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("zzz") is None

    def test_returns_none_when_index_missing(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") is None

    def test_returns_none_when_index_corrupt(self, tmp_path):
        (tmp_path / "sessions.json").write_text("NOT JSON", encoding="utf-8")
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") is None

    def test_returns_none_for_empty_title(self, tmp_path):
        self._write_index(tmp_path, [
            {"sessionId": "abc", "title": "   ", "messageCount": 1},
        ])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") is None

    def test_normalizes_newlines_in_title(self, tmp_path):
        self._write_index(tmp_path, [
            {"sessionId": "abc", "title": "Line one\r\nLine two\nLine three", "messageCount": 1},
        ])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") == "Line one Line two Line three"

    def test_truncates_long_title_with_ellipsis(self, tmp_path):
        long_title = "A" * 90
        self._write_index(tmp_path, [
            {"sessionId": "abc", "title": long_title, "messageCount": 1},
        ])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        result = adapter.get_session_title("abc")
        assert result == "A" * 80 + "..."

    def test_exact_80_char_title_has_no_ellipsis(self, tmp_path):
        title = "B" * 80
        self._write_index(tmp_path, [
            {"sessionId": "abc", "title": title, "messageCount": 1},
        ])
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") == title

    def test_returns_none_when_index_is_not_a_list(self, tmp_path):
        (tmp_path / "sessions.json").write_text(
            json.dumps({"sessionId": "abc", "title": "Oops"}), encoding="utf-8"
        )
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert adapter.get_session_title("abc") is None


# ---------------------------------------------------------------------------
# VSCodeCopilotAdapter.get_session_title
# ---------------------------------------------------------------------------

def _make_state_vscdb(ws_dir: Path, entries: dict[str, dict]) -> Path:
    """
    Write a minimal state.vscdb SQLite file with chat.ChatSessionStore.index
    populated from the given {session_id: metadata_dict} mapping.
    """
    import sqlite3
    db_path = ws_dir / "state.vscdb"
    ws_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE IF NOT EXISTS ItemTable (key TEXT PRIMARY KEY, value TEXT)"
    )
    payload = json.dumps({"version": 1, "entries": entries})
    con.execute(
        "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
        ("chat.ChatSessionStore.index", payload),
    )
    con.commit()
    con.close()
    return db_path


class TestVSCodeCopilotAdapterGetSessionTitle:
    def test_returns_title_from_vscdb(self, tmp_path):
        ws = tmp_path / "ws1"
        _make_state_vscdb(ws, {
            "session-aaa": {"sessionId": "session-aaa", "title": "Scene Audit", "lastMessageDate": 0},
        })
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_session_title("session-aaa") == "Scene Audit"

    def test_returns_none_for_unknown_session(self, tmp_path):
        ws = tmp_path / "ws1"
        _make_state_vscdb(ws, {
            "session-aaa": {"sessionId": "session-aaa", "title": "Something", "lastMessageDate": 0},
        })
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_session_title("session-zzz") is None

    def test_merges_across_multiple_workspaces(self, tmp_path):
        _make_state_vscdb(tmp_path / "ws1", {
            "session-aaa": {"title": "From WS1"},
        })
        _make_state_vscdb(tmp_path / "ws2", {
            "session-bbb": {"title": "From WS2"},
        })
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_session_title("session-aaa") == "From WS1"
        assert adapter.get_session_title("session-bbb") == "From WS2"

    def test_title_cache_built_once(self, tmp_path):
        ws = tmp_path / "ws1"
        _make_state_vscdb(ws, {
            "session-aaa": {"title": "Cached Title"},
        })
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter._title_cache is None
        adapter.get_session_title("session-aaa")
        assert adapter._title_cache is not None
        # Mutate the DB after first call - cache should not reflect the change.
        import sqlite3
        con = sqlite3.connect(str(ws / "state.vscdb"))
        new_payload = json.dumps({"version": 1, "entries": {
            "session-aaa": {"title": "Changed Title"},
        }})
        con.execute(
            "UPDATE ItemTable SET value=? WHERE key='chat.ChatSessionStore.index'",
            (new_payload,),
        )
        con.commit()
        con.close()
        assert adapter.get_session_title("session-aaa") == "Cached Title"

    def test_returns_none_when_no_vscdb_files(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_session_title("session-aaa") is None

    def test_skips_workspace_without_chat_key(self, tmp_path):
        import sqlite3
        ws = tmp_path / "ws1"
        ws.mkdir()
        db_path = ws / "state.vscdb"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO ItemTable VALUES ('some.other.key', 'irrelevant')")
        con.commit()
        con.close()
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_session_title("session-aaa") is None

    def test_skips_entry_with_blank_title(self, tmp_path):
        ws = tmp_path / "ws1"
        _make_state_vscdb(ws, {
            "session-aaa": {"title": "   "},
        })
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert adapter.get_session_title("session-aaa") is None

    def test_skips_corrupt_vscdb(self, tmp_path):
        ws = tmp_path / "ws1"
        ws.mkdir()
        (ws / "state.vscdb").write_bytes(b"this is not sqlite")
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        # Should not raise; just return None.
        assert adapter.get_session_title("session-aaa") is None


# ---------------------------------------------------------------------------
# ContinueAdapter.get_watch_dirs
# ---------------------------------------------------------------------------

class TestContinueAdapterGetWatchDirs:
    def test_returns_list_with_sessions_dir(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        result = adapter.get_watch_dirs()
        assert result == [tmp_path]

    def test_returns_list_type(self, tmp_path):
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        assert isinstance(adapter.get_watch_dirs(), list)

    def test_custom_sessions_dir(self, tmp_path):
        sessions_dir = tmp_path / "my_sessions"
        adapter = ContinueAdapter(sessions_dir=str(sessions_dir))
        assert adapter.get_watch_dirs() == [sessions_dir]


# ---------------------------------------------------------------------------
# ContinueAdapter.push_context
# ---------------------------------------------------------------------------

class TestContinueAdapterPushContext:
    def _make_cfg(self, workspace_id: str = "test") -> dict:
        return {"workspace_id": workspace_id}

    def test_first_sight_calls_inject_anchor(self, tmp_path, monkeypatch):
        """First-sight session (prev_count=0, no existing anchor) injects fresh anchor."""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        session_file = session_dir / "sess1.json"
        session_file.write_text("[]", encoding="utf-8")

        adapter = ContinueAdapter(sessions_dir=str(session_dir))

        injected = []
        monkeypatch.setattr("frontend_adapter_continue.catalog.load_catalog", lambda root: {})
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.check_anchor",
            lambda path: False,
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.inject_anchor",
            lambda path, cat, replace_existing: injected.append((path, replace_existing)),
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.rebuild_catalog_from_subjects",
            lambda subjects, root: {"subjects": []},
        )
        monkeypatch.setattr(
            "frontend_adapter_continue._subjects.load_subjects",
            lambda root: [],
        )

        messages = [{"role": "user", "content": "hello"}]
        adapter.push_context("sess1", messages, 0, self._make_cfg(), tmp_path)

        # First call should be the first-sight inject (replace_existing=False)
        assert any(not replace for _, replace in injected)

    def test_skips_anchor_when_present(self, tmp_path, monkeypatch):
        """When anchor is present, only the refresh inject is called."""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        session_file = session_dir / "sess2.json"
        session_file.write_text("[]", encoding="utf-8")

        adapter = ContinueAdapter(sessions_dir=str(session_dir))

        refresh_calls = []
        monkeypatch.setattr("frontend_adapter_continue.catalog.load_catalog", lambda root: {})
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.check_anchor",
            lambda path: True,
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.inject_anchor",
            lambda path, cat, replace_existing: refresh_calls.append(replace_existing),
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.rebuild_catalog_from_subjects",
            lambda subjects, root: {"subjects": []},
        )
        monkeypatch.setattr(
            "frontend_adapter_continue._subjects.load_subjects",
            lambda root: [],
        )

        messages = [{"role": "user", "content": "hello"}] * 3
        adapter.push_context("sess2", messages, 2, self._make_cfg(), tmp_path)

        # Only the refresh call should happen (replace_existing=True)
        assert refresh_calls == [True]

    def test_truncation_calls_restore_anchor(self, tmp_path, monkeypatch):
        """When message count drops, restore_anchor_after_truncation is called."""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        session_file = session_dir / "sess3.json"
        session_file.write_text("[]", encoding="utf-8")

        adapter = ContinueAdapter(sessions_dir=str(session_dir))

        restore_calls = []
        monkeypatch.setattr("frontend_adapter_continue.catalog.load_catalog", lambda root: {})
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.check_anchor",
            lambda path: False,
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.restore_anchor_after_truncation",
            lambda path, cat, summary_text: restore_calls.append(summary_text),
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.inject_anchor",
            lambda path, cat, replace_existing: None,
        )
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.rebuild_catalog_from_subjects",
            lambda subjects, root: {"subjects": []},
        )
        monkeypatch.setattr(
            "frontend_adapter_continue._subjects.load_subjects",
            lambda root: [],
        )
        monkeypatch.setattr(
            "frontend_adapter_continue._db.load_assignments",
            lambda session_id, root: {},
        )

        # prev_count=5 but all_messages only has 3 - truncation
        messages = [{"role": "user", "content": "a"}] * 3
        adapter.push_context("sess3", messages, 5, self._make_cfg(), tmp_path)

        assert len(restore_calls) == 1
        assert "truncation recovery" in restore_calls[0].lower()

    def test_noop_when_session_path_is_none(self, tmp_path, monkeypatch):
        """push_context does nothing if get_session_path returns None."""
        adapter = ContinueAdapter(sessions_dir=str(tmp_path))
        monkeypatch.setattr(adapter, "get_session_path", lambda sid: None)

        called = []
        monkeypatch.setattr(
            "frontend_adapter_continue.catalog.load_catalog",
            lambda root: called.append("load") or {},
        )
        adapter.push_context("s", [], 0, {}, tmp_path)
        assert called == []


# ---------------------------------------------------------------------------
# VSCodeCopilotAdapter.get_watch_dirs
# ---------------------------------------------------------------------------

class TestVSCodeCopilotAdapterGetWatchDirs:
    def test_returns_wsd_dir(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        result = adapter.get_watch_dirs()
        assert result == [tmp_path]

    def test_returns_list_type(self, tmp_path):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        assert isinstance(adapter.get_watch_dirs(), list)


# ---------------------------------------------------------------------------
# VSCodeCopilotAdapter.push_context
# ---------------------------------------------------------------------------

class TestVSCodeCopilotAdapterPushContext:
    def _make_cfg(self, enabled: bool = True, instructions_path: str = "") -> dict:
        return {
            "ghcp_adapter": {
                "enabled": enabled,
                "instructions_path": instructions_path,
                "injection_token_limit": 600,
            },
            "subagent_model": "test-model",
        }

    def test_noop_when_disabled(self, tmp_path, monkeypatch):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        called = []
        monkeypatch.setattr(
            "frontend_adapter_ghcp.build_backend_adapter",
            lambda cfg: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        cfg = self._make_cfg(enabled=False, instructions_path=str(tmp_path / "instructions.md"))
        adapter.push_context("s", [{"role": "user", "content": "hi"}], 0, cfg, tmp_path)
        # No error = no-op worked

    def test_noop_when_instructions_path_empty(self, tmp_path, monkeypatch):
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        cfg = self._make_cfg(enabled=True, instructions_path="")
        adapter.push_context("s", [{"role": "user", "content": "hi"}], 0, cfg, tmp_path)
        # No error = no-op worked

    def test_noop_when_no_messages(self, tmp_path):
        instructions = tmp_path / "instructions.md"
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        cfg = self._make_cfg(enabled=True, instructions_path=str(instructions))
        adapter.push_context("s", [], 0, cfg, tmp_path)
        assert not instructions.exists()

    def test_writes_fence_block_to_instructions(self, tmp_path, monkeypatch):
        instructions = tmp_path / "instructions.md"
        instructions.write_text("# My instructions\n", encoding="utf-8")

        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))

        class _FakeBackend:
            def complete(self, system_prompt, user_prompt, model):
                return "Active task: fix the bug."

        monkeypatch.setattr("frontend_adapter_ghcp.build_backend_adapter", lambda cfg: _FakeBackend())

        cfg = self._make_cfg(enabled=True, instructions_path=str(instructions))
        messages = [{"role": "user", "content": "What is the bug?"}]
        adapter.push_context("abcd1234", messages, 0, cfg, tmp_path)

        content = instructions.read_text(encoding="utf-8")
        assert "<!-- HUBRIS:START -->" in content
        assert "Active task: fix the bug." in content
        assert "<!-- HUBRIS:END -->" in content

    def test_replaces_existing_fence_block(self, tmp_path, monkeypatch):
        instructions = tmp_path / "instructions.md"
        instructions.write_text(
            "# Header\n<!-- HUBRIS:START -->\nOld content.\n<!-- HUBRIS:END -->\n# Footer\n",
            encoding="utf-8",
        )

        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))

        class _FakeBackend:
            def complete(self, system_prompt, user_prompt, model):
                return "New content."

        monkeypatch.setattr("frontend_adapter_ghcp.build_backend_adapter", lambda cfg: _FakeBackend())

        cfg = self._make_cfg(enabled=True, instructions_path=str(instructions))
        adapter.push_context("abcd5678", [{"role": "user", "content": "hi"}], 0, cfg, tmp_path)

        content = instructions.read_text(encoding="utf-8")
        assert "Old content." not in content
        assert "New content." in content
        assert "# Header" in content
        assert "# Footer" in content

    def test_noop_when_backend_returns_none(self, tmp_path, monkeypatch):
        instructions = tmp_path / "instructions.md"
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))

        class _FakeBackend:
            def complete(self, system_prompt, user_prompt, model):
                return None

        monkeypatch.setattr("frontend_adapter_ghcp.build_backend_adapter", lambda cfg: _FakeBackend())

        cfg = self._make_cfg(enabled=True, instructions_path=str(instructions))
        adapter.push_context("abcd0000", [{"role": "user", "content": "hi"}], 0, cfg, tmp_path)
        assert not instructions.exists()

    def test_noop_when_no_model_configured(self, tmp_path):
        instructions = tmp_path / "instructions.md"
        adapter = VSCodeCopilotAdapter(workspace_storage_dir=str(tmp_path))
        cfg = {
            "ghcp_adapter": {
                "enabled": True,
                "instructions_path": str(instructions),
                "injection_token_limit": 600,
            },
        }
        # No subagent_model or meta_model key - should not raise, just log and return
        adapter.push_context("abcd9999", [{"role": "user", "content": "hi"}], 0, cfg, tmp_path)
        assert not instructions.exists()
