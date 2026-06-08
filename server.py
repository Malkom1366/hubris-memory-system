"""
HuBrIS - Meta Cognitive MCP server.

Entry point and startup wiring. All MCP tool definitions live in tools.py.

An autonomous meta-cognitive memory layer for AI agent sessions.
Watches all Continue session files in the background and maintains a Dewey-tree
subject catalog with per-subject memory files.

Tools exposed to the agent (see tools.py for full list and disabled tools):
  recall_catalog   - browse all subjects with one-line previews (start here)
  recall_subject   - full structured memory for a known subject
  recall_vector    - semantic search when you have a concept but not a subject id
  recall_range     - chronological message retrieval across a UTC time window
  recall_exact     - retrieve a single message by its id (follow-up after vector/range)
  list_subjects    - tabular subject listing with states and counts
  declare_subject  - register a new named subject
  close_subject    - archive a subject and compact its session messages
  force_compact    - trigger an immediate lifecycle + synthesis pass
  get_status       - watcher status and config summary

The background watcher runs without any agent interaction. When a session
changes, it:
  1. Reads the delta since last check
  2. Classifies new messages into subjects
  3. Updates subject memory files
  4. Checks for catalog anchor (restores verbatim from catalog.json if missing)
  5. Refreshes the anchor in the session file

Run with: python server.py
Add to ~/.continue/config.json under "mcpServers".
"""

import sys
from pathlib import Path

import supervisor as _supervisor

import config as _config
import db as _db
from frontend_adapters import build_active_frontend_adapters
import log as _log_module
from log import get_logger

# Import the FastMCP instance from tools. This import also executes tools.py,
# registering all @mcp.tool() decorators before the server starts.
import tools as _tools
from tools import mcp  # noqa: F401 - re-exported for callers that do `from server import mcp`

_log = get_logger("hubris.server")

# Route FastMCP's own logger to hubris.log so tool-exception tracebacks appear
# there instead of being silently discarded (the mcp.* loggers have no handlers
# by default and do not propagate to root).
_log_module.attach_external_logger("mcp")

# ---------------------------------------------------------------------------
# Backward-compatibility re-exports.
#
# Tests and external callers that do `import server; server.recall_subject(...)`
# or `patch("server._db.X")` continue to work without changes because:
#  - Tool functions re-exported below are the same objects defined in tools.py.
#  - Module aliases (_config, _db, _emb, _subjects) point to the same module
#    objects that tools.py uses, so mock.patch("server._db.X") patches the
#    attribute on the shared db module and tools.py code sees the patch.
# ---------------------------------------------------------------------------

from tools import _config, _db, _emb, _subjects  # noqa: F401

from tools import (  # noqa: F401
    recall_catalog,
    recall_subject,
    recall_vector,
    recall_range,
    recall_exact,
    list_subjects,
    declare_subject,
    close_subject,
    move_subject,
    list_session_workspaces,
    get_status,
    set_config,
    force_compact,
)


# ---------------------------------------------------------------------------
# Startup argument parsing
# ---------------------------------------------------------------------------

def _parse_startup_args() -> bool:
    """
    Parse HuBrIS-specific CLI arguments from sys.argv.

    The only recognised flag is --configure, which forces the startup config
    dialog open.  Adapter and session overrides are not accepted here - they
    must be set through the config UI and saved before daemons launch.

    Returns configure (bool).
    """
    return _config.parse_startup_args(sys.argv[1:])



# ---------------------------------------------------------------------------
# Startup - only runs when executed directly, not when imported by tests.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import config_ui as _config_ui

    # Start MCP immediately so VS Code gets its `initialize` response and stops
    # retrying. Tools are gated by _runtime is None until config is confirmed.
    import threading as _threading
    _mcp_thread = _threading.Thread(target=mcp.run, daemon=True, name="hubris-mcp")
    _mcp_thread.start()

    _configure = _parse_startup_args()

    # Load the pre-existing config only to seed the UI with current values.
    # The authoritative config used by all daemons is what the user saves here.
    _cfg_seed = _config.load()

    # Show the startup config dialog on every run. The user's saved choices are
    # the exclusive source of truth for adapter, workspace_id, and all settings.
    _ui_result = _config_ui.show(_cfg_seed)
    if _ui_result is False:
        sys.exit(0)
    elif _ui_result is not None:
        _config.save(_ui_result)

    # Load config fresh from disk - this is exactly what the user just saved.
    _cfg = _config.load()

    _root_init = _config.memory_root(_cfg.get("workspace_id", "global"))
    _db.init_db(_root_init)
    _active_adapters = build_active_frontend_adapters(_cfg)
    for _install_adapter in _active_adapters:
        _install_adapter.install_rules(_cfg)

    if _supervisor.launch():
        _log.info("SUPERVISOR: primary - daemons launched")
    else:
        _log.warning("SUPERVISOR: secondary - MCP tools only (daemons managed by another process)")

    # Populate the tools module with runtime state now that startup is complete.
    _tools._runtime = _tools._HubrisRuntime(
        adapter=_active_adapters[0] if _active_adapters else None,
    )

    _mcp_thread.join()
