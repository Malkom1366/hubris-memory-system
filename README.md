# HuBrIS - Human Brain Inference Storage

Semi-autonomous background memory system. Allows the chat agent to query its own memory
with tools, while automated daemons perform tightly-bounded actions over the database layer.
Highly modular structure with per-daemon restarting behavior. Designed to be fault-resilient,
extensible, upgradeable, and inspectable across all surfaces.

---

## How It Works

HuBrIS runs as an MCP server. When your chat agent starts a session, HuBrIS is already
running in the background, watching session files for new messages and processing them.

The **autonomous tier** operates without any agent interaction:

| Daemon | Role |
|--------|------|
| `watcher` | Polls session files for new messages and enqueues them |
| `classify` | Assigns messages to semantic subjects using the subagent model |
| `embed` | Vectorizes messages for semantic search |
| `synthesize` | Compacts archived subjects into structured memory using the meta model |
| `split` | Detects large subjects that should be divided into focused children |
| `compact` | Reduces active session history when the token count exceeds the ceiling |
| `escalate` | Re-routes unclassifiable messages to the meta model |
| `reclass` | Re-attributes messages after a subject split |
| `attenuate` | Decays confidence scores on semantic links over time |
| `reconcile` | Detects agreement, contradiction, and supersession between memories |

Each daemon restarts independently on failure. The supervisor manages the full set and
exposes a consolidated health status through `get_status`.

The **voluntary tier** is the set of tools the chat agent calls explicitly - to browse
the subject catalog, retrieve memory, or trigger lifecycle actions when needed.

Memory persists in `~/.hubris/<workspace_id>/hubris.db` (SQLite, WAL mode) across
sessions and workspace moves. The default workspace_id is `global`, giving one
universal brain for all sessions unless you partition by project.

---

## Prerequisites

- Python 3.12 or later (3.14 tested)
- [Ollama](https://ollama.com) running locally
- The following models pulled in Ollama (see [Tested Models](#tested-models)):
  - A meta model (reasoning/synthesis)
  - A subagent model (classification)
  - An embedding model

---

## Installation

```bash
git clone https://github.com/Malkom1366/hubris-memory-system.git
cd hubris-memory-system
pip install -r requirements.txt
```

Copy the example config and edit it:

```bash
cp config.example.json config.json
```

Open `config.json` and fill in at minimum `meta_model`, `subagent_model`, and `embed_model`
with the names of models you have pulled in Ollama.

---

## Configuration

All fields are optional - the server writes defaults on first run. See `config.example.json`
for a complete template.

| Key | Default | Description |
|-----|---------|-------------|
| `backend_adapter` | `"ollama"` | Backend to use. Currently only `"ollama"` is formally supported. |
| `backends.ollama.endpoint` | `"http://localhost:11434"` | Ollama API endpoint. |
| `backends.ollama.timeout` | `600` | SSE read timeout in seconds. Set high enough to cover model cold-start. |
| `meta_model` | `"qwen3:30b-a3b"` | Reasoning model for synthesis, lifecycle decisions, and compaction. 27B+ quantized recommended. |
| `subagent_model` | `"qwen3:8b"` | Fast classification model. 3B-8B sufficient. |
| `embed_model` | `"nomic-embed-text"` | Embedding model for semantic recall. Set to `""` to disable vectorization. |
| `workspace_id` | `"global"` | Memory namespace. Change to isolate per-workspace. |
| `active_adapters` | `["continue"]` | Ordered list of session file adapters. Available: `"continue"`, `"ghcp"`. |
| `adapters.continue.sessions_dir` | `""` | Path to Continue sessions directory. Empty = auto-detected. |
| `adapters.ghcp.workspace_storage_dir` | `""` | Path to VS Code workspaceStorage. Empty = auto-detected. |
| `copilot_workspace_roots` | `[]` | Workspace root paths that receive a copy of `hubris-guide.md` at startup. |
| `poll_interval` | `2` | Seconds between watcher polls. |
| `max_active_tokens` | `8000` | Token ceiling before compact daemon triggers. |
| `split_subject_threshold` | `150` | Message count at which a subject becomes eligible for splitting. |
| `disabled_daemons` | `[]` | List of daemon names to skip at startup. E.g. `["reconcile", "attenuate"]`. |
| `finalize_max_chars` | `6000` | Max source characters included in the synthesis prompt. |

---

## Wiring Into VS Code Copilot Chat

Add HuBrIS as an MCP server in your VS Code `mcp.json`. This can be workspace-scoped
(`.vscode/mcp.json`) or user-scoped.

```json
{
  "servers": {
    "hubris": {
      "type": "stdio",
      "command": "python",
      "args": ["C:/path/to/hubris-memory-system/server.py"]
    }
  }
}
```

To give Copilot the agent operating guide, add the workspace root to `copilot_workspace_roots`
in `config.json`. HuBrIS will write `hubris-guide.md` into the `.github/` directory of
each listed root at server startup. The guide contains full tool descriptions and operating
instructions written for the agent.

---

## Wiring Into Continue.dev

Add to your `~/.continue/config.yaml`:

```yaml
mcpServers:
  - name: hubris
    command: python
    args:
      - C:/path/to/hubris-memory-system/server.py
```

Continue's session files are auto-detected from `~/.continue/sessions/`. Override with
`adapters.continue.sessions_dir` in `config.json` if your Continue installation is in
a non-standard location.

---

## Agent Guide

`hubris-guide.md` contains the full tool reference for the chat agent - what each tool
does, when to call it, and how to chain tools together. HuBrIS auto-deploys this file
to any workspace listed in `copilot_workspace_roots`. For Continue, wire it in as an
always-apply context provider or read it manually once when starting a new session.

---

## Tested Models

The following combinations have been validated to produce correct behavior:

| Role | Model |
|------|-------|
| Meta model | `qwen3:30b-a3b`, `qwen3.6:35b-a3b-q4_K_M` |
| Subagent model | `qwen3:8b`, `qwen3:4b` |
| Embed model | `nomic-embed-text` (8192-token context, 768-dim) |

HuBrIS uses standard Ollama chat and embed endpoints. Any model Ollama supports should
work in principle, provided it follows the chat completion response format. If you try
other models, the classify and synthesize behavior will vary with model quality - the
classify daemon has a threshold and blacklisting mechanism to protect against persistent
parse failures so runaway failures do not halt progress.

---

## Known Limitations

- **Windows only** - Tested on Windows. `Path.home()` resolves cross-platform, but
  adapter path auto-detection and default `sessions_dir` values assume Windows paths.
  Linux/macOS users are not supported in this release.

- **Ollama backend only** - The `backend_adapter_ollama.py` adapter is the only
  formally supported backend. The adapter interface is documented in `adapters.py`
  and `backend_adapters.py` for anyone who wants to implement additional backends.

- **Local storage only** - Memory lives in `~/.hubris/` as a local SQLite database.
  No cloud sync, no multi-machine sharing as of this release.

- **No automated retry for permanently blacklisted messages** - Messages that hit the
  classify failure threshold are blacklisted with a reason code. They can be manually
  unblocked via the database, but there is no automatic retry mechanism yet.

---

## License

[MIT](LICENSE) - Copyright (c) 2026 Libra Logic
