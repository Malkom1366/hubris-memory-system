# Release Notes

---

## v0.1.0 - May 30, 2026

Initial public release.

### What This Is

HuBrIS is a semi-autonomous background memory system that runs as an MCP server alongside
your chat agent. It watches session files written by Continue.dev and VS Code Copilot Chat,
classifies messages into a semantic subject catalog, and maintains structured long-term
memory the agent can query on demand.

The agent does not manage its own memory. It only reads from it when it needs to. All
indexing, synthesis, compaction, and archival happen in the background via a supervisor
managing ten independent daemons.

### What Is Included

- Ten background daemons: watcher, classify, embed, synthesize, split, compact, escalate,
  reclass, attenuate, reconcile
- MCP tool surface: `recall_catalog`, `recall_subject`, `recall_vector`, `recall_range`,
  `recall_exact`, `list_subjects`, `declare_subject`, `close_subject`, `force_compact`,
  `get_status`
- Ollama backend adapter with configurable endpoint and timeout
- Continue.dev and VS Code Copilot Chat session file adapters
- Dewey-tree subject catalog with open/dormant/archived lifecycle
- Semantic recall via vector similarity (requires embedding model)
- Chronological recall via UTC time window
- Agent operating guide (`hubris-guide.md`) with full tool reference
- Interactive configuration UI (`python config_ui.py`)
- Per-daemon restart behavior; daemon health tracked by supervisor
- Classify failure threshold and blacklisting with reason codes (timeout vs. content failure)
- Finalize bounded-message budget to prevent long GPU holds
- SQLite WAL-mode database at `~/.hubris/<workspace_id>/hubris.db`

### Tested Models

| Role | Models Validated |
|------|-----------------|
| Meta model | `qwen3:30b-a3b`, `qwen3.6:35b-a3b-q4_K_M` |
| Subagent model | `qwen3:8b`, `qwen3:4b` |
| Embed model | `nomic-embed-text` |

The meta model does the reasoning-heavy work: synthesis, lifecycle decisions, subject
splitting, and reconciliation. It needs genuine capability - 27B+ quantized is a
reasonable floor. The subagent model only classifies and confirms; a fast 4B-8B model
is sufficient. The embed model context window matters: `nomic-embed-text` has an 8192-token
window and handles dense conversation content without truncation errors.

Other Ollama-compatible models will work to the degree they follow the chat completion
response format and have sufficient reasoning quality. The system will adapt; expect
worse classification fidelity with weaker models.

### Known Limitations

- Windows only (tested). Cross-platform path handling is implemented but adapter
  auto-detection defaults assume Windows paths.
- Ollama backend only. The adapter interface (`adapters.py`, `backend_adapters.py`)
  is documented for third-party implementation.
- No automated retry for permanently blacklisted messages. Manual database intervention
  required to unblock entries with `status='blacklisted'`.
- No multi-machine memory sync. Storage is local SQLite only.

### Notes on Configuration

The embed model dimension is stored in the database schema. If you change `embed_model`
to a model with a different output dimension, you will need to drop and recreate the
vector tables. A migration utility will be provided in a future release; for now, the
database can be reset by deleting `~/.hubris/<workspace_id>/hubris.db`.

---

*Developed by Libra Logic.*
