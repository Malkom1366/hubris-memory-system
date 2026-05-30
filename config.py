"""
HuBrIS configuration management.

Config is stored alongside server.py in config.json.
Memory is stored in ~/.hubris/<workspace_id>/ so it survives project moves.
"""

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "config.json"
HUBRIS_HOME = Path.home() / ".hubris"

# Shared tiktoken encoding name used for token counting across the codebase.
# Both daemon_watcher.py and tools.py run in separate processes but must agree
# on this value, so it lives here as the single source of truth.
TOKEN_ENCODING_NAME = "cl100k_base"

DEFAULTS: dict[str, Any] = {
    # Active backend adapter name. Must match a ADAPTER_NAME in a
    # backend_adapter_<name>.py file. Only one backend is used at a time.
    "backend_adapter": "ollama",
    # Per-backend connection config. Keys vary by adapter.
    # ollama: endpoint, api_key, timeout (SSE read timeout in seconds)
    "backends": {
        "ollama": {
            "endpoint": "http://localhost:11434",
            "api_key": "",
            # SSE read timeout in seconds. Gap between consecutive chunks.
            # Set high enough to cover model cold-start (VRAM load).
            "timeout": 600,
        },
    },
    # Coordinating meta-agent: subject detection, lifecycle, compaction decisions.
    # Needs genuine reasoning. 27B+ quantized recommended.
    "meta_model": "qwen3:30b-a3b",
    # Mechanical subagent: dross confirmation, tagging, format tasks.
    # Fast small model sufficient. 3B-7B recommended.
    "subagent_model": "qwen3:8b",
    # Embedding model for semantic recall. Must be pulled in Ollama.
    # Set to "" to disable vectorization entirely.
    "embed_model": "nomic-embed-text",
    # Memory namespace. Change to isolate per-workspace. "global" gives a universal brain.
    "workspace_id": "global",
    # Ordered list of adapter names to activate. Adapters are polled in this order.
    # Available built-in names: "continue", "ghcp"
    "active_adapters": ["continue"],
    # Per-adapter configuration. Keys vary by adapter.
    # continue: sessions_dir (path to ~/.continue/sessions or override)
    # ghcp:     workspace_storage_dir (path to VS Code workspaceStorage, empty = auto)
    "adapters": {
        "continue": {"sessions_dir": ""},
        "ghcp": {"workspace_storage_dir": ""},
    },
    # List of workspace root paths whose .github/ directories should receive a copy of
    # hubris-guide.md at server startup. Add each workspace root that uses GitHub Copilot.
    # Example: ["C:\\Development\\my-project"]
    "copilot_workspace_roots": [],
    # GitHub Copilot Chat context injection. When enabled, HuBrIS distills recent
    # conversation messages and writes a HUBRIS fence block into the configured
    # copilot-instructions.md file after each classification cycle.
    "ghcp_adapter": {
        "enabled": False,
        "instructions_path": "",
        "injection_token_limit": 600,
    },
    # Watcher poll interval in seconds.
    "poll_interval": 2,
    # Token ceiling before compaction is triggered for a session.
    "max_active_tokens": 8000,
    # Token budget for a single recall tool response. Responses exceeding
    # this are truncated with a guidance note so the agent can re-request
    # with tighter parameters.
    "max_recall_tokens": 6000,
    # Message count at which a subject is eligible for splitting into child subjects.
    # Must be an integer >= 1. Set to 0 to disable automatic split detection.
    "split_subject_threshold": 150,
    # Recency weighting for recall_vector. Boosts similarity scores for more recently
    # linked messages using: final_score = similarity * (1 + alpha * exp(-lambda * delta))
    # where delta is the age of the semantic link in days.
    # Set recency_alpha to 0.0 to disable recency weighting entirely.
    "recency_alpha": 0.3,
    "recency_lambda": 0.005,
    # Daemon names to skip at startup. Any name matching a "name" field in manifest.json
    # will not be launched by the supervisor. Use the --configure dialog to manage this.
    # Example: ["reconcile", "attenuate"] disables those two daemons.
    "disabled_daemons": [],
    # daemon_reconcile: relation detection between memories sharing the same subject.
    # How often the scanner fires (seconds).
    "reconcile_interval_s": 60,
    # How many semantically-similar neighbor candidates to retrieve per link.
    "reconcile_candidates_k": 5,
    # Additive confidence adjustments applied by the LLM judgment pass.
    #   supports   - new memory corroborates the older one; boost old confidence
    #   contradicts - genuine conflict; reduce old confidence
    #   updates    - old fact superseded; reduce old confidence more sharply
    "reconcile_delta_supports": 0.05,
    "reconcile_delta_contradicts": -0.10,
    "reconcile_delta_updates": -0.20,
    # Maximum characters of ranked source messages included in the finalize prompt.
    # Messages are appended in descending (confidence, recency) order until this
    # limit is reached. Lower values yield shorter LLM calls; raise when using a
    # model that handles longer context comfortably.
    "finalize_max_chars": 6000,
}

def memory_root(workspace_id: str | None = None) -> Path:
    """Return the memory root for the given workspace_id (or the one in config)."""
    if workspace_id is None:
        workspace_id = load().get("workspace_id", "global")
    root = HUBRIS_HOME / workspace_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save(DEFAULTS.copy())
        return DEFAULTS.copy()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = DEFAULTS.copy()
    merged.update(data)
    return merged


def save(cfg: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def update(**kwargs: Any) -> dict[str, Any]:
    """Update zero or more config fields. Ignores unknown keys and None values."""
    cfg = load()
    for key, value in kwargs.items():
        if key in DEFAULTS and value is not None:
            cfg[key] = value
    save(cfg)
    return cfg


def parse_startup_args(argv: list[str]) -> tuple[str | None, str | None, bool]:
    """
    Parse HuBrIS-specific CLI arguments.

    Recognised flags (order-independent, case-insensitive):
      --adapter:<name>   Override config.json 'adapter' key ('copilot' or 'continue').
      --session:<id>     Bind this instance to one specific session ID.
      --configure        Open the startup config dialog before launching daemons.
                         Also opens automatically when config.json is absent.

    Returns (adapter_override, session_id, configure).
    """
    adapter_override: str | None = None
    session_id: str | None = None
    configure: bool = False
    for arg in argv:
        lower = arg.lower()
        if lower.startswith("--adapter:"):
            adapter_override = arg.split(":", 1)[1].strip()
        elif lower.startswith("--session:"):
            session_id = arg.split(":", 1)[1].strip()
        elif lower == "--configure":
            configure = True
    return adapter_override, session_id, configure
