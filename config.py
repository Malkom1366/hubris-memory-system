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
            # How long (seconds) to wait for the inference backend to send its
            # first response byte. Covers any queue wait in front of the backend
            # plus cold-start time (e.g. loading a large model into VRAM).
            # Set this high enough that legitimate queue waits never time out.
            "start_timeout": 600,
            # Per-chunk read timeout (seconds) once the model is generating.
            # Each SSE token resets this clock. Only fires if the model stalls
            # mid-generation. 90s is generous for any real stall scenario.
            "timeout": 90,
        },
    },
    # Coordinating meta-agent: subject detection, lifecycle, compaction decisions.
    # Needs genuine reasoning. 27B+ quantized recommended.
    "meta_model": "qwen3:30b-a3b",
    # Mechanical subagent: dross confirmation, tagging, format tasks.
    # Fast small model sufficient. 3B-7B recommended.
    "subagent_model": "qwen3:8b",
    # ---------------------------------------------------------------------------
    # Context window sizes (tokens). These are the primary knobs that control
    # ALL derived size budgets below. Set each to the actual context window of
    # the model you are running. When you upgrade hardware and load a larger
    # model, raise the corresponding value here and all budgets scale up
    # automatically without touching anything else.
    #
    # How to find the right value:
    #   - Ollama model page / README (e.g. "phi4-reasoning-8k" -> 8192)
    #   - `ollama show <model>` -> look for "context length" in the output
    #   - When in doubt, use a conservative (smaller) number - the system
    #     degrades gracefully; it just truncates more aggressively.
    # ---------------------------------------------------------------------------
    # Subagent model context window (tokens). Drives: content_max_chars.
    # Set to the context length of your subagent_model.
    # Find it with: ollama show <model> | grep context
    "subagent_context_tokens": 4096,
    # Meta model context window (tokens). Drives: finalize_max_chars, escalation_token_max.
    # Set to the context length of your meta_model. Can be larger than subagent if your
    # meta model supports it; default is conservative for modest hardware.
    "meta_context_tokens": 4096,
    # Embedding model context (tokens). Drives: embed_chunk_size, embed_chunk_overlap.
    # nomic-embed-text default in Ollama is 2048. mxbai-embed-large is also 512-dim
    # but check its context; set higher if your model supports it.
    "embed_context_tokens": 2048,
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
    # Computed automatically from meta_context_tokens by compute_budgets().
    # Override here only if you want a fixed cap regardless of context size.
    # "finalize_max_chars": 10384,
}

# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------

def compute_budgets(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Derive all LLM prompt size limits from model-specific context window sizes,
    filling derived keys into cfg only when they are absent.

    Each model's budget is computed independently so prompts never exceed that
    model's context window. Explicit values already in cfg are never overwritten.

    Derivation rationale (all token counts assume 4 chars per token):

    content_max_chars  (subagent - total message block budget)
      Reserve ~1500 tokens for system prompt, structured fields, and response.
      The remaining tokens are available for message content. Callers divide
      this proportionally across however many messages are in the prompt, so
      total content always stays within budget regardless of message count.
      => (sa_ctx - 1500) * 4  chars

    finalize_max_chars  (meta - total message block budget for finalize/summarize)
      Same reservation formula, derived from meta model's context.
      => (mt_ctx - 1500) * 4  chars

    escalation_token_max  (meta - input token budget for escalation prompt)
      Reserve 600 tokens for system prompt, subjects list, and response.
      => mt_ctx - 600  tokens

    embed_chunk_size  (embed model - chars per chunk)
      65% of embed context. Keeps 35% headroom so the Ollama
      "exceeds context length" 400 error cannot fire.
      => emb_ctx * 65 // 100 * 4  chars

    embed_chunk_overlap  (overlap between adjacent embed chunks)
      10% of chunk_size preserves boundary context.
      => embed_chunk_size // 10  chars
    """
    sa_ctx  = int(cfg.get("subagent_context_tokens", 4096))
    mt_ctx  = int(cfg.get("meta_context_tokens", 4096))
    emb_ctx = int(cfg.get("embed_context_tokens", 2048))

    out = dict(cfg)

    if "content_max_chars" not in cfg:
        out["content_max_chars"] = max(500, (sa_ctx - 1500) * 4)

    if "finalize_max_chars" not in cfg:
        out["finalize_max_chars"] = max(1000, (mt_ctx - 1500) * 4)

    if "escalation_token_max" not in cfg:
        out["escalation_token_max"] = max(500, mt_ctx - 600)

    if "embed_chunk_size" not in cfg:
        out["embed_chunk_size"] = max(500, emb_ctx * 65 // 100 * 4)

    if "embed_chunk_overlap" not in cfg:
        chunk_size = int(out.get("embed_chunk_size", 5324))
        out["embed_chunk_overlap"] = max(50, chunk_size // 10)

    return out

_WINDOWS_FILENAME_INVALID = frozenset('\\/:*?"<>|')


def sanitize_workspace_id(workspace_id: str) -> str:
    """
    Return a version of workspace_id that is safe to use as a single directory
    component on Windows (and POSIX).

    All characters that Windows forbids in file/directory names are replaced with
    underscores.  Every other character - including ampersands, hashes, spaces,
    percent signs, exclamation marks, etc. - is preserved as-is because Windows
    allows them in file names and they carry user intent.

    Also strips leading/trailing whitespace and dots (Windows ignores trailing
    dots in directory names, which causes silent path mismatches).

    A blank result after sanitization falls back to "global".
    """
    sanitized = "".join("_" if c in _WINDOWS_FILENAME_INVALID else c for c in workspace_id)
    sanitized = sanitized.strip().strip(".")
    return sanitized or "global"


def memory_root(workspace_id: str | None = None) -> Path:
    """Return the memory root for the given workspace_id (or the one in config)."""
    if workspace_id is None:
        workspace_id = load().get("workspace_id", "global")
    root = HUBRIS_HOME / sanitize_workspace_id(workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save(DEFAULTS.copy())
        return compute_budgets(DEFAULTS.copy())
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = DEFAULTS.copy()
    merged.update(data)
    return compute_budgets(merged)


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


def parse_startup_args(argv: list[str]) -> bool:
    """
    Parse HuBrIS-specific CLI arguments.

    The only recognised flag is --configure, which forces the startup config
    dialog open even when config.json already exists.  Adapter selection and
    session binding are not accepted as CLI flags - they must be set through
    the config UI and saved to config.json before the daemons launch.

    Returns configure (bool).
    """
    configure: bool = False
    for arg in argv:
        if arg.lower() == "--configure":
            configure = True
    return configure
