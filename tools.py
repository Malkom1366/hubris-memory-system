"""
HuBrIS MCP tool definitions.

All tool definitions exposed to the agent via MCP. The FastMCP instance (mcp)
is created here and imported by server.py, which populates runtime state
via a single _HubrisRuntime instance (_runtime) after startup.

Active tools (registered with FastMCP):
  recall_catalog           - browse all subjects with one-line previews (start here)
  recall_subject           - full structured memory for a known subject
  recall_vector            - semantic search when you have a concept but not a subject id
  recall_range             - chronological message retrieval across a UTC time window
  recall_exact             - retrieve a single message by its id (follow-up after vector/range)
  list_subjects            - tabular subject listing with states and counts
  declare_subject          - register a new named subject
  close_subject            - archive a subject and retain its memory
  move_subject             - move a subject to a different parent in the hierarchy
  get_status               - watcher status and config summary
  force_compact            - trigger an immediate lifecycle + synthesis pass

Disabled tools (decorator commented out - not exposed to the agent):
  list_session_workspaces  - list all workspaces with Copilot/Continue sessions
  set_config               - update config fields at runtime
"""

import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import tiktoken
from mcp.server.fastmcp import FastMCP
from pydantic import Field

import catalog
import config as _config
import db as _db
import embeddings as _emb
import subjects as _subjects
from daemon_split import _maybe_enqueue_split
from frontend_adapters import SessionAdapter
from log import get_logger

mcp = FastMCP("hubris")
_log = get_logger("hubris.server")

# ---------------------------------------------------------------------------
# Runtime state - populated by server.py after startup
# ---------------------------------------------------------------------------

@dataclass
class _HubrisRuntime:
    """Container for daemon-layer state injected by server.py at startup."""
    adapter: Any                        # SessionAdapter instance (primary frontend adapter)
    bound_session: str | None           # if set, only this session is processed

_runtime: _HubrisRuntime | None = None

_STARTUP_PENDING = (
    "HuBrIS is not yet ready - the startup configuration dialog is still open. "
    "Please complete it in VS Code."
)


def _assert_ready() -> str | None:
    """Return an error string if startup is not yet complete, else None."""
    return _STARTUP_PENDING if _runtime is None else None


def _read_daemon_watcher_status() -> dict[str, Any]:
    """Read the watcher status file written by daemon_watcher.py after each poll cycle.

    Falls back to a default dict if the file does not yet exist or cannot be read.
    The status file is written to _config.HUBRIS_HOME / 'watcher_status.json'.
    """
    _default: dict[str, Any] = {
        "running": False,
        "session_count": 0,
        "watching": [],
        "in_flight": [],
        "queued": [],
        "poll_interval": 2,
    }
    status_path = _config.HUBRIS_HOME / "watcher_status.json"
    try:
        return json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return _default

# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

_TOKEN_ENCODING = tiktoken.get_encoding(_config.TOKEN_ENCODING_NAME)
_BUDGET_TRUNCATION_NOTE = (
    "\n\n[RESULT TRUNCATED: This response exceeded the recall token budget. "
    "Re-request with tighter parameters (narrower time range, fewer results, "
    "or a specific subject id) to receive complete content.]"
)


def _guard_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens (cl100k_base) and append a guidance note."""
    tokens = _TOKEN_ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _TOKEN_ENCODING.decode(tokens[:max_tokens])
    return truncated + _BUDGET_TRUNCATION_NOTE


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def recall_catalog() -> str:
    """
    Browse the complete subject catalog with one-line memory previews.

    Returns every registered subject with its Dewey id, state, and the first
    line of its memory file. Use this as the entry point when you do not already
    know which subject id to look up.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    all_subjects = _subjects.list_subjects_summary(root)
    if not all_subjects:
        return "No subjects registered yet."
    lines = ["HuBrIS Subject Catalog", ""]
    for s in all_subjects:
        mem = _subjects.read_subject_memory(s, root)
        preview = ""
        if mem:
            view = _subjects.parse_memory_view(mem)
            preview_source = view if view else mem
            first_line = (
                preview_source.strip().splitlines()[0]
                if preview_source.strip()
                else ""
            )
            preview = f" | {first_line[:120]}" if first_line else ""
        lines.append(
            f"  [{s['dewey_id']}] {s['name']} ({s['state']}){preview}"
        )
    result = "\n".join(lines)
    return _guard_token_budget(result, int(cfg.get("max_recall_tokens", 6000)))


@mcp.tool()
def recall_subject(
    id: Annotated[
        str,
        Field(
            description=(
                "Subject id (short hex id) or Dewey id (e.g. '0', '0.1') to retrieve."
            ),
        ),
    ],
) -> str:
    """
    Retrieve full structured memory for a known subject.

    Returns the synthesized memory file content for the subject, formatted for
    reading. Use recall_catalog to find ids, or recall_vector when you have a
    concept but not an exact id.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    subject = _subjects.get_subject(id, root)
    if subject is None:
        return f"Subject '{id}' not found."
    mem = _subjects.read_subject_memory(subject, root)
    if not mem:
        return f"Subject '{subject['name']}' exists but has no memory content yet."
    view = _subjects.parse_memory_view(mem)
    content = view if view else mem
    result = f"## {subject['name']} [{subject['dewey_id']}]\n\n{content}"
    # Append belief history if relation edges exist for this subject.
    try:
        edges = _db.get_subject_relations_for_subject(subject["id"], root)
        if edges:
            n = len(edges)
            label = "edge" if n == 1 else "edges"
            lines = [f"\n\n### Belief history ({n} {label})"]
            for e in edges:
                lines.append(
                    f"- link {e['from_link_id']} (conf {e['from_confidence']:.2f})"
                    f" --{e['relation']}--> link {e['to_link_id']}"
                    f" (conf {e['to_confidence']:.2f})"
                    f"  [{e['created_date_utc'][:10]}]"
                )
            result += "\n".join(lines)
    except Exception as exc:
        _log.warning("recall_subject: relation graph query failed: %s", exc)
    # Recall detector: surface split eligibility so oversized subjects get
    # partitioned even if the lifecycle watchdog hasn't fired yet.
    try:
        _maybe_enqueue_split(subject, root, cfg)
    except Exception as exc:
        _log.warning("recall_subject split-check failed: %s", exc)
    return _guard_token_budget(result, int(cfg.get("max_recall_tokens", 6000)))


@mcp.tool()
def recall_vector(
    query: Annotated[
        str,
        Field(description="Natural language query to search autobiographical memory."),
    ],
    k: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            le=50,
            description="Number of results to return (1-50, default 10).",
        ),
    ] = 10,
) -> str:
    """
    Semantic memory search using vector similarity.

    Embeds the query and returns the k most similar messages from
    autobiographical memory, ranked by cosine distance. Each result includes
    the subject context, speaker, distance score, message id, and a content
    preview. Use recall_exact with a message id to retrieve the full content
    of any result.

    Requires sqlite-vec to be installed and embed_model to be configured.
    Returns a descriptive error if the embedding stack is not available.
    """
    if msg := _assert_ready():
        return msg
    if not _emb.is_available():
        return (
            "Vectorization is not enabled. "
            "Install sqlite-vec to use this tool: pip install sqlite-vec"
        )
    cfg = _config.load()
    model = cfg.get("embed_model", "").strip()
    if not model:
        return (
            "Vectorization is not enabled. "
            'Set embed_model in config.json, for example: "embed_model": "mxbai-embed-large"'
        )
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    embedding_bytes = _emb.embed(query, cfg)
    if embedding_bytes is None:
        return (
            f"Could not embed the query using model '{model}'. "
            "Check that Ollama is running and the model is pulled: "
            f"ollama pull {model}"
        )
    msg_results = _db.semantic_search_messages(embedding_bytes, k, root)
    subject_results = _db.semantic_search_subjects(embedding_bytes, k, root)

    if not msg_results and not subject_results:
        return (
            "No similar results found. "
            "Memory may not be vectorized yet - try again after the next watcher cycle."
        )

    alpha = float(cfg.get("recency_alpha", 0.3))
    lam = float(cfg.get("recency_lambda", 0.005))

    # Score message results (optionally boosted by recency).
    now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    for row in msg_results:
        distance = float(row.get("distance") or 1.0)
        similarity = max(0.0, 1.0 - distance)
        if alpha > 0.0:
            link_date_str = row.get("link_created_date")
            if link_date_str:
                try:
                    link_dt = datetime.datetime.fromisoformat(link_date_str)
                    delta_days = max(0.0, (now_utc - link_dt).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    delta_days = 0.0
            else:
                delta_days = 0.0
            row["_score"] = similarity * (1.0 + alpha * math.exp(-lam * delta_days))
        else:
            row["_score"] = similarity
        row["_result_type"] = "message"

    # Score subject results (no recency boost - subject memory is synthetic).
    for row in subject_results:
        distance = float(row.get("distance") or 1.0)
        row["_score"] = max(0.0, 1.0 - distance)
        row["_result_type"] = "subject"

    # Merge and rank.
    all_results = sorted(msg_results + subject_results, key=lambda r: r["_score"], reverse=True)

    lines = [f"Semantic recall for: {query!r}", f"Top {len(all_results)} result(s)", ""]
    for rank, row in enumerate(all_results, 1):
        result_type = row["_result_type"]
        score = float(row.get("_score") or 0.0)
        if result_type == "subject":
            subject_name = row.get("name") or "(unnamed)"
            dewey_id = row.get("dewey_id") or ""
            content = str(row.get("memory_content") or "")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            lines.append(
                f"[{rank}] type=SUBJECT subject={subject_name!r} dewey={dewey_id} score={score:.4f}"
            )
            lines.append(f"    {preview}")
        else:
            subject_name = row.get("subject_name") or "(unclassified)"
            speaker = str(row.get("speaker") or "unknown")
            content = str(row.get("raw_content") or "")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            session_id = str(row.get("session_id") or "")[:8]
            msg_idx = row.get("message_index")
            msg_id = str(row.get("id") or "")
            lines.append(
                f"[{rank}] type=MESSAGE subject={subject_name!r} speaker={speaker} "
                f"score={score:.4f} session={session_id} idx={msg_idx} id={msg_id}"
            )
            lines.append(f"    {preview}")
        lines.append("")
    result = "\n".join(lines)
    return _guard_token_budget(result, int(cfg.get("max_recall_tokens", 6000)))


@mcp.tool()
def recall_range(
    start_utc: Annotated[
        str,
        Field(
            description=(
                "Start of the time window as an ISO 8601 UTC string, "
                "e.g. '2026-05-05T00:00:00'. Inclusive."
            ),
        ),
    ],
    end_utc: Annotated[
        str,
        Field(
            description=(
                "End of the time window as an ISO 8601 UTC string, "
                "e.g. '2026-05-12T23:59:59'. Inclusive."
            ),
        ),
    ],
    session_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional - narrow results to a single session id.",
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(
            default=100,
            ge=1,
            le=500,
            description="Maximum number of messages to return (1-500, default 100).",
        ),
    ] = 100,
) -> str:
    """
    Retrieve messages chronologically across a UTC time window.

    Use this when you have a fuzzy time reference ('three weeks ago when we
    implemented X') and need to translate it into a concrete list of messages.
    Results are ordered by timestamp then message index. Use session_id to
    narrow to a specific conversation. Use recall_exact with a message id to
    retrieve any result's full content.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    rows = _db.get_messages_in_range(start_utc, end_utc, session_id, limit, root)
    if not rows:
        return f"No messages found between {start_utc} and {end_utc}."
    lines = [
        f"Time range: {start_utc} to {end_utc}",
        f"{len(rows)} message(s) found"
        + (f" (showing first {limit}, narrow the range to see more)" if len(rows) == limit else ""),
        "",
    ]
    for row in rows:
        ts = str(row.get("timestamp_utc") or "")
        sid = str(row.get("session_id") or "")[:8]
        idx = row.get("message_index")
        speaker = str(row.get("speaker") or "unknown")
        subject_name = row.get("subject_name") or "(unclassified)"
        msg_id = str(row.get("id") or "")
        content = str(row.get("raw_content") or "")
        preview = content[:300] + ("..." if len(content) > 300 else "")
        lines.append(
            f"[{ts}] session={sid} idx={idx} id={msg_id} "
            f"subject={subject_name!r} speaker={speaker}"
        )
        lines.append(f"  {preview}")
        lines.append("")
    result = "\n".join(lines)
    return _guard_token_budget(result, int(cfg.get("max_recall_tokens", 6000)))


@mcp.tool()
def recall_exact(
    message_id: Annotated[
        str,
        Field(
            description=(
                "The id of the message to retrieve, as returned by recall_vector "
                "or recall_range."
            ),
        ),
    ],
) -> str:
    """
    Retrieve a single message by its id.

    Returns the full message content with all subject associations. Use this
    after recall_vector or recall_range to drill into a specific result whose
    preview was truncated.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    row = _db.get_message_by_id(message_id, root)
    if row is None:
        return f"Message '{message_id}' not found."
    ts = str(row.get("timestamp_utc") or "")
    session_id = str(row.get("session_id") or "")
    idx = row.get("message_index")
    speaker = str(row.get("speaker") or "unknown")
    subject_names = str(row.get("subject_names") or "(unclassified)")
    content = str(row.get("raw_content") or "")
    result = (
        f"Message id={message_id}\n"
        f"Session: {session_id} | Index: {idx} | Timestamp: {ts}\n"
        f"Speaker: {speaker}\n"
        f"Subjects: {subject_names}\n\n"
        f"{content}"
    )
    return _guard_token_budget(result, int(cfg.get("max_recall_tokens", 6000)))


@mcp.tool()
def list_subjects() -> str:
    """
    List all registered subjects with their Dewey IDs, states, and message counts.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    all_subjects = _subjects.list_subjects_summary(root)
    if not all_subjects:
        return "No subjects registered yet."
    lines = ["Subject catalog:"]
    for s in all_subjects:
        desc = f" - {s['description']}" if s.get("description") else ""
        lines.append(
            f"  {s['dewey_id']}  {s['name']} [{s['state']}] "
            f"({s['message_count']} messages){desc}"
        )
    return "\n".join(lines)


@mcp.tool()
def declare_subject(
    name: Annotated[str, Field(description="Short name for the subject (e.g. 'database-migration').")],
    description: Annotated[
        str,
        Field(default="", description="One-sentence description of what this subject covers."),
    ] = "",
    parent_id: Annotated[
        str | None,
        Field(default=None, description="Dewey id of the parent subject, if this is a sub-topic."),
    ] = None,
) -> str:
    """
    Register a new named subject in HuBrIS's memory catalog.
    Future messages will be classified against this subject during background processing.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    try:
        subject = _subjects.create_subject(
            name=name,
            description=description,
            parent_id=parent_id,
            root=root,
        )
        _log.info(
            "SUBJECT DECLARED name=%r dewey=%s id=%s",
            subject["name"], subject["dewey_id"], subject["id"],
        )
        # Rebuild and re-inject the catalog anchor into all active sessions.
        all_subjects = _subjects.load_subjects(root)
        rebuilt = catalog.rebuild_catalog_from_subjects(all_subjects, root)
        catalog.inject_anchor_all_sessions(_runtime.adapter, rebuilt)
        return (
            f"Subject '{subject['name']}' registered "
            f"(id: {subject['id']}, dewey: {subject['dewey_id']})."
        )
    except ValueError as e:
        return str(e)


# ---------------------------------------------------------------------------
# close_subject MCP tool
# ---------------------------------------------------------------------------


@mcp.tool()
def close_subject(
    id: Annotated[str, Field(description="Short hex id or Dewey id of the subject to archive.")],
) -> str:
    """
    Archive a subject: flip its state to archived and enqueue a Phase 2
    finalization pass. No session-file rewrite, no autobiographical-memory
    mutation. The accumulated memory is retained verbatim and remains
    accessible via recall(subject_name).
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    # Resolve the subject.
    all_subjects = _subjects.load_subjects(root)
    subject = next(
        (s for s in all_subjects if s["id"] == id or s["dewey_id"] == id),
        None,
    )
    if subject is None:
        return f"Subject '{id}' not found."

    subject_id = subject["id"]
    subject_name = subject["name"]
    _subjects.archive_subject(subject, root)

    # Rebuild and broadcast the catalog anchor so all sessions reflect the new state.
    rebuilt = catalog.rebuild_catalog_from_subjects(_subjects.load_subjects(root), root)
    catalog.inject_anchor_all_sessions(_runtime.adapter, rebuilt)

    return (
        f"Subject '{subject_name}' archived. Memory retained; "
        f"use recall('{subject_id}') to retrieve."
    )


@mcp.tool()
def move_subject(
    subject_id: Annotated[str, Field(description="Short hex id or Dewey id of the subject to move.")],
    new_parent_id: Annotated[
        str | None,
        Field(description="Short hex id or Dewey id of the new parent, or null to make the subject top-level."),
    ] = None,
) -> str:
    """
    Move a subject to a different parent in the hierarchy (or to top-level).
    Reassigns the subject's Dewey ID and updates all descendant Dewey IDs accordingly.
    Use this to fix incorrect parent assignments or reorganize the subject tree.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    try:
        result = _subjects.move_subject(subject_id, new_parent_id, root)
    except (KeyError, ValueError) as exc:
        return f"Error: {exc}"
    parent_desc = f"under '{result['new_parent_id']}'" if result["new_parent_id"] else "top-level"
    desc_note = (
        f" ({result['descendants_updated']} descendant(s) also renumbered)"
        if result["descendants_updated"]
        else ""
    )
    return (
        f"Moved '{result['name']}': {result['old_dewey']} -> {result['new_dewey']} "
        f"({parent_desc}){desc_note}."
    )


# @mcp.tool()  # commented out - workspace inspection via chat agent is disabled
def list_session_workspaces() -> str:
    """
    List all workspaces that have Copilot or Continue sessions on this machine.

    For the Copilot adapter: shows each workspaceStorage hash alongside its
    decoded human-readable path (read from VS Code's workspace.json), the number
    of transcript files, and whether the workspace is currently in the blacklist.
    Use this to discover hashes to add to watched_workspace_hashes in config.json.

    For the Continue adapter: shows all unique workspaceDirectory values seen
    across known sessions.  Use this to discover paths for watched_workspaces.
    """
    from pathlib import Path as _Path
    import json as _json
    import os as _os

    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    # --- Copilot adapter path ---
    if hasattr(_runtime.adapter, "_wsd"):
        storage_root = _Path(_runtime.adapter._wsd)
        if not storage_root.is_dir():
            return f"workspaceStorage not found at: {storage_root}"

        workspace_bl = _db.load_workspace_blacklist(root)
        lines = ["Copilot workspace sessions (workspaceStorage):"]
        lines.append(f"  {'HASH':<36}  {'SESSIONS':>8}  {'BL':>3}  PATH")
        lines.append("  " + "-" * 90)

        for ws_dir in sorted(storage_root.iterdir()):
            if not ws_dir.is_dir():
                continue
            transcripts_dir = ws_dir / "GitHub.copilot-chat" / "transcripts"
            if not transcripts_dir.is_dir():
                continue
            session_count = sum(1 for f in transcripts_dir.glob("*.jsonl"))
            hash_name = ws_dir.name

            # Resolve human-readable path from workspace.json
            wjson = ws_dir / "workspace.json"
            decoded_path = "(unknown)"
            if wjson.is_file():
                try:
                    meta = _json.loads(wjson.read_text(encoding="utf-8"))
                    raw_uri = meta.get("folder") or meta.get("workspace") or ""
                    decoded_path = catalog.decode_workspace_uri(raw_uri) if raw_uri else "(no folder)"
                except Exception:
                    decoded_path = "(unreadable)"

            bl_flag = "YES" if hash_name in workspace_bl else " no"
            lines.append(f"  {hash_name:<36}  {session_count:>8}  {bl_flag:>3}  {decoded_path}")

        watched_hashes = cfg.get("watched_workspace_hashes", [])
        if watched_hashes:
            lines.append(f"\nwatched_workspace_hashes: {watched_hashes}")
        else:
            lines.append("\nwatched_workspace_hashes not set - all Copilot workspaces allowed (except blacklisted).")
        return "\n".join(lines)

    # --- Continue adapter path ---
    _watch_dirs = _runtime.adapter.get_watch_dirs()
    sessions_dir = _watch_dirs[0] if _watch_dirs else None
    if sessions_dir is None:
        return "Adapter does not expose session workspace information."

    workspaces: dict[str, int] = {}
    for f in _Path(sessions_dir).glob("*.json"):
        if f.name == "sessions.json":
            continue
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                raw = data.get("workspaceDirectory") or "(none)"
            else:
                raw = "(legacy bare-list format)"
        except Exception:
            raw = "(unreadable)"
        decoded = catalog.decode_workspace_uri(raw) if raw not in ("(none)", "(legacy bare-list format)", "(unreadable)") else raw
        workspaces[decoded] = workspaces.get(decoded, 0) + 1

    if not workspaces:
        return "No Continue sessions found."

    lines = ["Workspace directories seen across all Continue sessions:"]
    for ws, count in sorted(workspaces.items()):
        lines.append(f"  [{count:3d} session(s)]  {ws}")
    watched = cfg.get("watched_workspaces", [])
    if watched:
        lines.append(f"\nCurrently watched: {watched}")
    else:
        lines.append("\nwatched_workspaces is not configured - all workspaces are processed.")
    return "\n".join(lines)


@mcp.tool()
def get_status() -> str:
    """
    Return HuBrIS watcher status, active session count, config summary,
    and memory root location.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))
    status = _read_daemon_watcher_status()
    all_subjects = _subjects.list_subjects_summary(root)
    open_count = sum(1 for s in all_subjects if s["state"] == "open")
    dormant_count = sum(1 for s in all_subjects if s["state"] == "dormant")
    bl_sessions = len(_db.load_session_blacklist(root))
    bl_msg_ranges = len(_db.load_message_blacklist(root))
    pending_failures = len(_db.load_classify_failures(root))

    # Estimate token usage for the calling session only.
    # _runtime.bound_session is set at startup via the CLI --session:<id> flag.
    max_tokens = int(cfg.get("max_active_tokens", 8000))
    active_sid = _runtime.bound_session

    ctx_block: list[str]
    if active_sid and active_sid in status["watching"]:
        try:
            msgs = _runtime.adapter.read_messages(active_sid)
            chars = sum(
                len(m["content"])
                for m in msgs
                if not m.get("content", "").lstrip().startswith(catalog.ANCHOR_MARKER)
            )
            tok = chars // 4
        except Exception:
            tok = 0
        bar_width = 16
        filled = min(bar_width, round(bar_width * tok / max_tokens)) if max_tokens > 0 else 0
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        pct = round(100 * tok / max_tokens) if max_tokens > 0 else 0
        flag = " (!)" if tok >= max_tokens else ""
        ctx_block = [f"    {active_sid[:8]}  {tok:>6,} / {max_tokens:,}  {bar}  {pct}%{flag}"]
    else:
        ctx_block = ["    (session not yet identified - send a message to activate tracking)"]
    lines = [
        "HuBrIS Status",
        "",
        f"  Watcher running:    {status['running']}",
        f"  Sessions watched:   {status['session_count']}",
        f"  In-flight cycles:   {len(status['in_flight'])}",
        f"  Poll interval:      {status['poll_interval']}s",
        "",
        f"  Session binding:    {'explicit (' + _runtime.bound_session[:8] + ')' if _runtime.bound_session else 'whitelist'}",
        f"  Adapter:            {cfg.get('adapter', 'continue')}",
        "",
        f"  Memory root:        {root}",
        f"  Subjects (open):    {open_count}",
        f"  Subjects (dormant): {dormant_count}",
        "",
        f"  Blacklisted sessions:     {bl_sessions}",
        f"  Blacklisted msg ranges:   {bl_msg_ranges} session(s)",
        f"  Pending failure counters: {pending_failures}",
        "",
        f"  Context usage (est. tokens, ceiling {max_tokens:,}):",
        *ctx_block,
        "",
        f"  Meta model:         {cfg.get('meta_model', '(not set)')}",
        f"  Subagent model:     {cfg.get('subagent_model', '(not set)')}",
        f"  Endpoint:           {cfg.get('backends', {}).get('ollama', {}).get('endpoint', '(not set)')}",
    ]
    return "\n".join(lines)


# @mcp.tool()  # commented out - config mutation via chat agent is disabled
def set_config(
    backend_endpoint: Annotated[str | None, Field(default=None, description="Backend endpoint URL (e.g. http://localhost:11434).")] = None,
    meta_model: Annotated[str | None, Field(default=None, description="Model name for the meta-agent tier.")] = None,
    subagent_model: Annotated[str | None, Field(default=None, description="Model name for the subagent tier.")] = None,
    workspace_id: Annotated[str | None, Field(default=None, description="Memory namespace (default: 'global').")] = None,
    poll_interval: Annotated[float | None, Field(default=None, description="Watcher poll interval in seconds.")] = None,
    max_active_tokens: Annotated[int | None, Field(default=None, description="Token ceiling before compaction.")] = None,
) -> str:
    """
    Update HuBrIS configuration at runtime. Only provided fields are changed.
    The new config is saved to config.json immediately.
    """
    cfg = _config.load()
    changed_parts: list[str] = []

    if backend_endpoint is not None:
        cfg.setdefault("backends", {}).setdefault("ollama", {})["endpoint"] = backend_endpoint
        changed_parts.append(f"backend_endpoint={backend_endpoint!r}")
    if meta_model is not None:
        cfg["meta_model"] = meta_model
        changed_parts.append(f"meta_model={meta_model!r}")
    if subagent_model is not None:
        cfg["subagent_model"] = subagent_model
        changed_parts.append(f"subagent_model={subagent_model!r}")
    if workspace_id is not None:
        cfg["workspace_id"] = workspace_id
        changed_parts.append(f"workspace_id={workspace_id!r}")
    if poll_interval is not None:
        cfg["poll_interval"] = poll_interval
        # TODO: IPC to daemon_watcher to apply new poll_interval at runtime
        changed_parts.append(f"poll_interval={poll_interval!r}")
    if max_active_tokens is not None:
        cfg["max_active_tokens"] = max_active_tokens
        changed_parts.append(f"max_active_tokens={max_active_tokens!r}")

    if not changed_parts:
        return "No fields provided. Nothing changed."

    _config.save(cfg)
    return f"Config updated: {', '.join(changed_parts)}"


@mcp.tool()
def force_compact() -> str:
    """
    Trigger an immediate compaction cycle regardless of normal thresholds.

    - Runs _check_dormant_subjects now, promoting any stale open subjects to
      dormant and archiving any subjects that have exceeded the archive threshold.
    - Drains the entire Phase 2 synthesis queue (all pending archived subjects,
      not just one).
    - Returns a summary of what was processed.
    """
    if msg := _assert_ready():
        return msg
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    promoted = 0
    archived = 0

    # Lifecycle pass - delegates to subjects.run_lifecycle_pass so the threshold
    # logic lives in exactly one place.
    promoted, archived = _subjects.run_lifecycle_pass(root, cfg, _runtime.adapter)

    # daemon_synthesize handles the memory-actions queue asynchronously.

    parts = []
    if promoted:
        parts.append(f"{promoted} subject(s) promoted to dormant")
    if archived:
        parts.append(f"{archived} subject(s) auto-archived")
    if not parts:
        parts.append("nothing to compact - all subjects current")
    return "force_compact: " + "; ".join(parts) + "."
