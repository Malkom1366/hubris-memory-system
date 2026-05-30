"""
Front-end session adapter framework for HuBrIS.

Public entry point for all front-end adapter usage. Defines the SessionAdapter
protocol and the build_frontend_adapter / build_active_frontend_adapters factories.
Concrete adapter implementations live in their own modules:

  frontend_adapter_continue  - ContinueAdapter
  frontend_adapter_ghcp      - VSCodeCopilotAdapter

External callers import from this module only; the concrete adapter modules
are implementation details. Auto-discovery and multi-adapter construction are
handled by registry.py.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from frontend_adapter_continue import ContinueAdapter as ContinueAdapter
from frontend_adapter_ghcp import VSCodeCopilotAdapter as VSCodeCopilotAdapter
from registry import build_active_frontend_adapters as build_active_frontend_adapters

__all__ = [
    "SessionAdapter",
    "ContinueAdapter",
    "VSCodeCopilotAdapter",
    "build_frontend_adapter",
    "build_active_frontend_adapters",
]


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

    def get_watch_dirs(self) -> list[Path]:
        """Return filesystem directories this adapter monitors for session changes."""
        ...

    def push_context(
        self,
        session_id: str,
        all_messages: list[dict],
        prev_count: int,
        cfg: dict,
        root: Path,
    ) -> None:
        """
        Post-classification context push. Each adapter handles its own unique
        context injection behavior:
          ContinueAdapter:      anchor check, truncation recovery, anchor refresh
          VSCodeCopilotAdapter: distill recent messages, write HUBRIS fence block
        Called after classification has been enqueued.
        """
        ...

    def install_rules(self, cfg: dict) -> None:
        """
        Install any adapter-specific guide files into locations the host IDE
        reads at startup. Called once at server startup. Must be idempotent - if
        the destination already has identical content it should be a no-op.
        """
        ...

    def workspace_id_allowed(self, workspace_id: str | None, cfg: dict) -> bool:
        """
        Return True if workspace_id passes the adapter's configured workspace
        filter (watched_workspaces for Continue, watched_workspace_hashes for
        Copilot). Returns True when no filter is configured.
        """
        ...

    def get_session_title(self, session_id: str) -> str | None:
        """
        Return a short human-readable title for the session, or None if no
        title can be determined.

        ContinueAdapter reads the title from the sessions.json index.
        VSCodeCopilotAdapter derives it from the first user.message in the
        JSONL transcript.
        """
        ...


def build_frontend_adapter(cfg: dict) -> SessionAdapter:
    """
    Return the single active front-end session adapter.

    Convenience shim over build_active_frontend_adapters. Returns the first
    adapter in the active list. Raises ValueError if the active list is empty.
    """
    adapters = build_active_frontend_adapters(cfg)
    if not adapters:
        raise ValueError("No active frontend adapters configured.")
    return adapters[0]
