---
alwaysApply: true
---
# HuBrIS - Human Brain Inference Storage - Operating Guide

This session has the HuBrIS toolset available. HuBrIS is an autonomous memory layer -
it classifies messages, updates subject memory, and injects catalog anchors into the
session file in the background without any per-turn agent action.

There is no end-of-turn call. There is no manual compaction. The only time you touch
tools is when you want to read memory, register a subject, or check status.

---

## Tool Reference

### Memory Retrieval

- `recall_catalog()` - No arguments. Lists the full subject catalog with one-line
  previews of each subject's memory content. Start here when you want to know what
  HuBrIS remembers about the current workspace.

- `recall_subject(id)` - Retrieve the full structured memory for a known subject.
  Pass a short hex id or Dewey id (e.g. `"0.1"`). Use `recall_catalog` first if you
  do not know the id.

- `recall_vector(query, k=10)` - Semantic search over autobiographical memory using
  vector similarity. Pass a natural language description of what you are looking for.
  Returns the k most similar messages with subject context, speaker, distance score,
  message id, and a content preview. Use `recall_exact` to drill into any result whose
  preview was truncated. Requires `embed_model` to be configured and Ollama running.

- `recall_range(start_utc, end_utc, session_id=None, limit=100)` - Retrieve messages
  chronologically across a UTC time window. Use when you have a fuzzy time reference
  ("three weeks ago when we implemented X") and need to translate it into concrete
  messages. Both timestamps are ISO 8601 strings (e.g. `"2026-05-05T00:00:00"`).
  Optional `session_id` narrows to a single conversation. Results include message ids
  for chaining to `recall_exact`.

- `recall_exact(message_id)` - Retrieve a single message by its id. Use after
  `recall_vector` or `recall_range` to read the full content of a result whose preview
  was truncated. The id is included in the output of both search tools.

- `list_subjects()` - Lists all subjects with Dewey IDs, states (open/dormant), and
  message counts. Use when you want the catalog in tabular form without memory previews.

### Subject Management

- `declare_subject(name, description="", parent_id=None)` - Register a new named
  subject. Use when the conversation shifts to a clearly distinct topic. `parent_id`
  accepts a Dewey id to nest under an existing subject.

- `close_subject(id)` - Archive a subject: compacts all messages assigned to it in
  active session files into a single placeholder turn, marks the subject archived, and
  queues a final synthesis pass. Use when a topic is fully resolved. Memory remains
  accessible via `recall_catalog` and `recall_subject`.

- `force_compact()` - No arguments. Runs a full lifecycle pass immediately (promotes
  stale open subjects to dormant, archives dormant subjects past the threshold) and
  finalizes memory consolidation for any recently archived subjects. Use when you want
  compaction to happen now rather than waiting for the background watcher cycle.

### Diagnostics and Config

- `get_status` - Watcher state, active session count, subject counts, memory root,
  and model config. Use when something seems wrong or the user asks about session state.

- `set_config(...)` - Update fields at runtime: `endpoint`, `meta_model`,
  `subagent_model`, `embed_model`, `workspace_id`, `poll_interval`,
  `max_active_tokens`, `max_recall_tokens`, `dormant_after_minutes`,
  `archive_after_minutes`.
  Only supplied fields change. Changes persist to config.json immediately.

---

## What runs automatically (no tool call needed)

The background watcher handles all of this without agent involvement:
- Detects session file changes every ~2s
- Classifies new messages into registered subjects
- Updates per-subject memory files
- Embeds messages for semantic recall (requires `embed_model` configured)
- Injects/refreshes the catalog anchor in the session file
- Detects context truncation and restores the anchor with subject recall pointers
- Promotes open subjects to dormant after `dormant_after_minutes` of inactivity (default 60)
- Archives dormant subjects to long-term memory after `archive_after_minutes` of inactivity (default 120)
- Consolidates archived subject memory in the background after archival

---

## When to use tools

- **Start of a new distinct topic:** `declare_subject` - gives the watcher a target
  to classify messages into.
- **User asks what you remember:** `recall_catalog` for an overview, `recall_subject`
  for a specific topic.
- **Looking for something by concept:** `recall_vector` - describe it in natural language.
- **Looking for something by time:** `recall_range` - supply a UTC window.
- **Need the full text of a specific message:** `recall_exact` with the message id.
- **Topic is done:** `close_subject` to archive it and compact its session messages.
- **Want compaction now:** `force_compact()` - runs the full lifecycle pass immediately.
- **Something seems wrong with memory:** `get_status` to check watcher state.

All recall tools enforce a token budget (default 6000 tokens). If a result is truncated
you will see a `[RESULT TRUNCATED]` note - re-request with tighter parameters (narrower
time range, lower k, or a specific subject id).

Do not call any of these speculatively on every turn. The background loop runs
whether or not you interact with it.

---

## Hard rules

- **Never directly read or edit files under `~/.hubris/`.** Those files are owned
  exclusively by the HuBrIS background process. Direct edits will corrupt memory state
  or be silently overwritten. Use `recall_catalog`, `recall_subject`, `declare_subject`,
  `close_subject`, and `force_compact` for all memory interaction.
- **Never directly read or edit Continue session files under `~/.continue/sessions/`.**
  HuBrIS manages those files. Direct edits will corrupt the session anchor and counts.