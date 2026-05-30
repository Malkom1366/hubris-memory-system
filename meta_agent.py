"""
Meta-agent inference for HuBrIS.

All LLM calls go through this module. Two model tiers:
  - meta_model: coordinating agent (complex reasoning, subject classification,
    compaction decisions, summary generation). 27B+ recommended.
  - subagent_model: mechanical tasks (dross confirmation, format normalization,
    tag extraction). 3-7B sufficient.

The endpoint must be OpenAI-compatible (Ollama, LM Studio, etc.).
"""

import json
import re
import time
from typing import Any, Callable

import httpx

import config as _config
from backend_adapters import build_backend_adapter
from log import get_logger

_log = get_logger("hubris.agent")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """You are HuBrIS, a meta-cognitive memory layer for AI agent sessions.
Your job is to read new conversation messages and decide, for each one, which
known subjects it touches and how strongly it touches them.

You will be given:
  - A list of known subjects (id, name, description)
  - A list of new messages (role, content)

A single message often touches more than one subject. Include every subject the
message genuinely relates to, even briefly. Each subject for a message gets a
relevance score in [0.0, 1.0]:
  - 1.0 = the message is centrally about this subject
  - 0.5 = the message touches this subject as a meaningful side topic
  - 0.2 = the message brushes against this subject only in passing
  - below 0.2 = do not include the subject at all
The message's \"primary\" subject is implicitly the one with the highest relevance.

Respond with ONLY a JSON object mapping each 0-based message index to an object
of {subject_id: relevance}. A message that belongs to no known subject maps to
an empty object {}.

Example response:
{
  "0": {"a1b2c3d4": 1.0},
  "1": {"a1b2c3d4": 0.8, "e5f6g7h8": 0.4},
  "2": {},
  "3": {"e5f6g7h8": 1.0, "a1b2c3d4": 0.3, "i9j0k1l2": 0.25}
}

Rules:
- Only use subject ids from the provided list. Drop any others.
- Relevance must be a number in [0.0, 1.0].
- Multiple subjects per message are normal and expected.
- Respond with ONLY the JSON object. No explanation, no markdown fencing.
- If the message list is empty, respond with {}.
"""

_SUMMARIZE_SYSTEM_PROMPT = """You are HuBrIS, a meta-cognitive memory layer for AI agent sessions.
Your job is to generate a dense technical summary of a conversation block for a named subject.

The summary must:
- Preserve all decisions, file names, function names, error codes, and key facts.
- Be written as dense prose, not bullet points.
- Be concise: capture every useful fact in as few words as possible.
- Omit filler, greetings, and conversational scaffolding entirely.

Respond with ONLY the summary text. No labels, no markdown fencing.
"""

_DROSS_SYSTEM_PROMPT = """You are HuBrIS, a meta-cognitive memory layer for AI agent sessions.
Your job is to identify zero-information messages in a conversation.

A zero-information message is one that contributes no facts, decisions, or technical content.
Examples: "OK", "Got it", "Thanks", "Sure, I can do that", "Let me know if you need anything".

You will be given a list of messages (index, role, content).
Respond with a JSON array of the 0-based indices of zero-information messages.

Example response: [0, 3, 7]

Rules:
- Respond with ONLY the JSON array. No explanation, no markdown fencing.
- If no zero-information messages are found, respond with [].
"""

_FINALIZE_MEMORY_SYSTEM_PROMPT = """You are HuBrIS's memory archivist. A subject has been permanently closed. Your job is to produce its definitive, final memory record.

You receive:
1. The current materialized view (an incremental accumulation from per-turn updates - may be redundant or fragmented)
2. Source messages for this subject, ranked by relevance then recency (most relevant and most recent first; may be a bounded subset of the full history)

Produce a JSON object with exactly two keys:
- "view": the definitive materialized view in markdown - a single coherent, non-redundant summary that reflects everything the event log records. Resolve any contradictions in favor of the most recent entries. Remove redundant restatements. Organize with headings where the volume warrants it. This is the final canonical record that recall() will return forever.
- "log_entry": 1-2 sentences describing that this is a finalization synthesis pass (e.g. "Final synthesis on archive: consolidated N update entries into definitive summary.").

Rules for the view:
- Every key fact, decision, file name, function name, design choice, and conclusion from the log must be preserved
- Prefer structured notes over prose where possible
- Omit greetings, filler, and zero-information content entirely
- Do NOT invent information not present in the log or existing view

Respond with the JSON object only. No markdown fencing, no preamble.
"""

_SPLIT_SUBJECT_SYSTEM_PROMPT = """You are HuBrIS's memory taxonomist. A subject has grown too large and covers multiple distinct topics. Your job is to partition it into coherent child subjects.

You receive:
1. The parent subject name
2. The current materialized view
3. The full event log

Produce a JSON object with exactly one key:
- "children": a JSON array of 2 to 5 objects, each with:
  - "name": a concise subject name (3-6 words, no punctuation)
  - "view": a markdown summary containing only the facts from the parent that belong to this child

Rules:
- Every fact from the parent view and log must appear in exactly one child. No fact may be dropped.
- Children must be genuinely distinct topics, not arbitrary slices. Do not split a single cohesive topic just to fill the quota.
- Prefer 2-3 children unless the content clearly demands more.
- Never produce more than 5 children.
- Do not invent information not present in the source.
- Child names must be usable as display labels and must not duplicate the parent name.

Respond with the JSON object only. No markdown fencing, no preamble.
"""

_UPDATE_MEMORY_SYSTEM_PROMPT = """You are HuBrIS's memory writer. You maintain a persistent memory file per subject across sessions.

You receive:
1. The current materialized view for the subject (empty if this is the first write)
2. New conversation messages assigned to this subject. Each message is prefixed
   with its role and a relevance score in [0.0, 1.0] for this subject. Higher
   relevance means the message is more central to the subject; lower-relevance
   turns provide context but should contribute less to the synthesis.

Produce a JSON object with exactly these three keys:
- "view": updated materialized view in markdown - dense and structured, preserving all technical
  facts (file names, function names, error messages, decisions, design patterns). Integrate new
  information without duplication. Remove conclusions that the new messages have superseded.
- "log_entry": 1-3 sentences in plain text describing what was added or changed in this update.
- "status": one of "established" (first write for this subject), "extended" (new information
  added, nothing overturned), or "revised" (prior conclusions corrected or significantly changed).

Rules for the view:
- Use markdown headings where volume warrants it
- Prefer structured notes over prose where possible
- Omit greetings, filler, and zero-information content entirely

Respond with the JSON object only. No markdown fencing, no preamble.
"""


# ---------------------------------------------------------------------------
# Low-level inference dispatch
# ---------------------------------------------------------------------------

def _call_llm(
    system_prompt: str,
    user_prompt: str,
    cfg: dict[str, Any],
    model_key: str = "meta_model",
) -> str:
    """Dispatch to the configured backend adapter and return assembled content."""
    model = cfg.get(model_key, cfg.get("meta_model", ""))
    return build_backend_adapter(cfg).complete(system_prompt, user_prompt, model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string using the standard 4-chars-per-token
    approximation. Accurate enough for threshold comparisons; no tokenizer
    dependency required.
    """
    return max(1, len(text) // 4)


class ClassifyFailed(Exception):
    """Raised when the LLM classify call fails and no assignments can be made."""

    def __init__(self, message: str, *, is_transient: bool = False) -> None:
        super().__init__(message)
        self.is_transient = is_transient


# Messages per classify LLM call. Classify is fully async so there is no
# cost to smaller batches. Small batches stay well within any model context
# window and limit retry surface when a batch fails.
# Overridable via config key "classify_batch_size".
_CLASSIFY_BATCH_SIZE = 10


def _classify_batch(
    batch: list[dict],
    batch_start: int,
    subjects_text: str,
    cfg: dict[str, Any],
) -> dict[int, dict[str, float]]:
    """
    Classify a single batch of messages. Returns a dict keyed by GLOBAL index
    (batch_start + local index), whose value is a {subject_id: relevance}
    map (possibly empty). Raises ClassifyFailed on any error.
    """
    messages_text = json.dumps(
        [{"index": i, "role": m["role"], "content": m["content"][:500]}
         for i, m in enumerate(batch)],
        ensure_ascii=False,
    )
    user_prompt = (
        f"Known subjects:\n{subjects_text}\n\n"
        f"New messages:\n{messages_text}"
    )
    try:
        raw = _call_llm(_CLASSIFY_SYSTEM_PROMPT, user_prompt, cfg, model_key="subagent_model")
        try:
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
        except json.JSONDecodeError as jde:
            _log.warning("CLASSIFY batch raw response (first 800 chars): %r", raw[:800])
            raise jde
        out: dict[int, dict[str, float]] = {}
        for k, v in parsed.items():
            global_idx = int(k) + batch_start
            entry: dict[str, float] = {}
            # Backwards-compatible parse: tolerate the old single-subject
            # shape (string or null) by mapping it to a relevance of 1.0.
            if v is None:
                entry = {}
            elif isinstance(v, str):
                entry = {v: 1.0}
            elif isinstance(v, dict):
                for sid, score in v.items():
                    try:
                        entry[str(sid)] = float(score)
                    except (TypeError, ValueError):
                        continue
            out[global_idx] = entry
        return out
    except httpx.HTTPError as e:
        _log.warning(
            "CLASSIFY batch at offset %d failed (%s) -> raising ClassifyFailed (transient)",
            batch_start, type(e).__name__,
        )
        raise ClassifyFailed(str(e), is_transient=True) from e
    except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as e:
        _log.warning(
            "CLASSIFY batch at offset %d failed (%s) -> raising ClassifyFailed",
            batch_start, type(e).__name__,
        )
        raise ClassifyFailed(str(e), is_transient=False) from e


def classify_messages(
    messages_delta: list[dict],
    known_subjects: list[dict],
    cfg: dict[str, Any] | None = None,
    *,
    on_batch_done: Callable[[dict[int, dict[str, float]], int], None] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Assign each message in messages_delta to zero or more known subjects with
    a relevance score in [0.0, 1.0] per (message, subject) pair.

    Processes messages in small batches so the prompt never exceeds the model
    context window. Returns a dict mapping 0-based global index ->
    {subject_id: relevance}. An empty inner dict means the message belongs to
    no known subject. Raises ClassifyFailed on any LLM or parse error.

    on_batch_done: optional callback invoked after each successful batch.
      Signature: on_batch_done(batch_relevance, messages_cumulative)
      - batch_relevance: dict mapping global index -> {subject_id: relevance}
      - messages_cumulative: total messages processed so far in this delta
      This enables per-batch committed checkpointing in the caller so a
      restart only retries from the last batch boundary, not from zero.
    """
    if cfg is None:
        cfg = _config.load()
    if not messages_delta:
        return {}

    subjects_text = json.dumps(
        [{"id": s["id"], "name": s["name"], "description": s.get("description", "")}
         for s in known_subjects],
        ensure_ascii=False,
    )
    batch_size = max(1, int(cfg.get("classify_batch_size", _CLASSIFY_BATCH_SIZE)))
    batches = [
        messages_delta[i:i + batch_size]
        for i in range(0, len(messages_delta), batch_size)
    ]
    _log.info(
        "CLASSIFY %d msg(s) against %d subject(s) in %d batch(es) of %d -> subagent_model=%s",
        len(messages_delta), len(known_subjects), len(batches), batch_size,
        cfg.get("subagent_model", "?"),
    )

    valid_ids: set[str] = {s["id"] for s in known_subjects}

    result: dict[int, dict[str, float]] = {}
    for batch_idx, batch in enumerate(batches):
        batch_start = batch_idx * batch_size
        batch_result = _classify_batch(batch, batch_start, subjects_text, cfg)
        # Validate: drop unknown subject ids; clamp scores into [0, 1].
        cleaned_count = 0
        invalid_count = 0
        for k, pair_map in batch_result.items():
            cleaned: dict[str, float] = {}
            for sid, score in pair_map.items():
                if sid not in valid_ids:
                    invalid_count += 1
                    continue
                cleaned[sid] = max(0.0, min(1.0, float(score)))
            batch_result[k] = cleaned
            cleaned_count += 1
        if invalid_count:
            _log.warning(
                "CLASSIFY batch %d: %d invalid subject id(s) cleared",
                batch_idx + 1, invalid_count,
            )
        result.update(batch_result)
        assigned = sum(1 for pm in batch_result.values() if pm)
        _log.info(
            "CLASSIFY batch %d/%d: %d/%d assigned",
            batch_idx + 1, len(batches), assigned, len(batch),
        )
        if on_batch_done is not None:
            on_batch_done(batch_result, batch_start + len(batch))

    total_assigned = sum(1 for pm in result.values() if pm)
    _log.info("CLASSIFY total: %d/%d assigned", total_assigned, len(messages_delta))
    return result


# ---------------------------------------------------------------------------
# Subject escalation: meta-model discovers and defines subjects
# ---------------------------------------------------------------------------

_SUBJECT_ESCALATION_SYSTEM_PROMPT = """\
You are HuBrIS, a meta-cognitive memory layer for AI assistant sessions.
A lightweight model attempted to classify the following messages against existing subjects
and could not find a match.

Re-examine each message carefully. The lightweight model may simply have missed the correct
assignment - start by checking whether any existing subject already covers this content.

Your tasks:
1. If a message belongs to an existing subject (even loosely), assign it there.
2. For messages that genuinely do not fit any existing subject, decide:
   - Should they start a new standalone subject?
   - Should they become a subsection (sub-topic) of an existing subject?
   - Are they dross or noise that should not be tracked?

Respond with ONLY a JSON object in this exact shape:
{
  "assignments": {"<index>": "<existing_subject_id_or_null>", ...},
  "new_subjects": [
    {"name": "...", "description": "...", "parent_id": "<subject_id_or_null>", "message_indices": [<int>, ...]}
  ]
}

Rules:
- "assignments" maps each message index to an existing subject id, or null if the message
  belongs to a new subject listed in "new_subjects" or is genuine dross.
- Every message index that appears in "new_subjects" must be null in "assignments".
- A message not in "new_subjects" and null in "assignments" is left unclassified.
- Do NOT create a new subject if an existing one already covers the topic - assign to it.
Parent assignment rules:
- Set parent_id ONLY when the new subject is a focused sub-aspect of an existing named domain.
  The parent must logically CONTAIN the child - not merely co-occur with it in the same session.
  Example of wrong parenting: "Docker" is NOT a child of "Python" because it is a container
  platform, not a Python library - even when both topics appear in a session about deploying a Python app.
- Do NOT make something a child because both topics appeared in the same conversation, or because
  the same developer works on both. Co-occurrence is not containment.
- When in doubt, set parent_id to null (top-level) rather than wrongly nest.

Granularity rules:
- Prefer narrow, focused subjects over broad catch-all buckets.
- If an existing subject already has many messages (50+) and new content fits a distinct
  sub-area of that domain, create a child subject for that sub-area instead of continuing to
  pile content into the parent.
- Component-level work (individual plugins, modules, subsystems) should be their own subject
  under their product parent, not merged into the parent product subject.
- name: 3-8 words, title case; description: 1-2 sentences describing what the subject covers.
- Include ALL message indices in "assignments".
- Respond with ONLY the JSON. No explanation, no markdown fencing."""


class EscalateFailed(Exception):
    """Raised when the meta-model escalation call fails and no assignments can be made."""


def escalate_unassigned(
    messages: list[dict],
    known_subjects: list[dict],
    cfg: dict[str, Any] | None = None,
) -> dict:
    """
    Escalate messages the subagent could not classify to the meta-model.

    The meta-model first checks whether any existing subject was missed (the subagent
    may simply have failed to recognise the connection), then proposes new subjects or
    subsections for content that genuinely does not fit anything already defined.

    Returns a dict:
        {
            "assignments": {local_int: subject_id_or_null, ...},
            "new_subjects": [
                {"name": str, "description": str,
                 "parent_id": str | None, "message_indices": [int, ...]}
            ]
        }
    Local indices are 0-based into `messages`. Raises EscalateFailed on LLM
    or parse error so the caller can decide whether to retry.
    """
    if cfg is None:
        cfg = _config.load()
    if not messages:
        return {"assignments": {}, "new_subjects": []}

    subjects_text = json.dumps(
        [{"id": s["id"], "name": s["name"], "description": s.get("description", "")}
         for s in known_subjects],
        ensure_ascii=False,
    )
    messages_text = json.dumps(
        [{"index": i, "role": m["role"], "content": m["content"][:500]}
         for i, m in enumerate(messages)],
        ensure_ascii=False,
    )
    user_prompt = (
        f"Existing subjects:\n{subjects_text}\n\n"
        f"Unmatched messages:\n{messages_text}"
    )
    _log.info(
        "ESCALATE %d unmatched msg(s) against %d subject(s) -> meta_model=%s",
        len(messages), len(known_subjects), cfg.get("meta_model", "?"),
    )
    try:
        raw = _call_llm(
            _SUBJECT_ESCALATION_SYSTEM_PROMPT, user_prompt, cfg, model_key="meta_model"
        )
        try:
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
        except json.JSONDecodeError as jde:
            _log.warning("ESCALATE raw response (first 800 chars): %r", raw[:800])
            raise jde
        assignments = {int(k): v for k, v in parsed.get("assignments", {}).items()}
        # Ensure every index is present; missing ones default to null.
        for i in range(len(messages)):
            if i not in assignments:
                assignments[i] = None
        new_subjects = parsed.get("new_subjects", [])
        reassigned = sum(1 for v in assignments.values() if v is not None)
        _log.info(
            "ESCALATE result: %d reassigned to existing, %d new subject(s) proposed",
            reassigned, len(new_subjects),
        )
        return {"assignments": assignments, "new_subjects": new_subjects}
    except (json.JSONDecodeError, ValueError, httpx.HTTPError, KeyError, IndexError, TypeError, AttributeError) as e:
        _log.warning(
            "ESCALATE failed (%s) -> raising EscalateFailed", type(e).__name__
        )
        raise EscalateFailed(str(e)) from e


def generate_subject_summary(
    messages: list[dict],
    subject_name: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """
    Generate a dense summary of messages for a named subject.
    Returns the summary string, or a fallback message on failure.
    """
    if cfg is None:
        cfg = _config.load()
    if not messages:
        return ""

    block_text = "\n\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in messages
    )
    user_prompt = (
        f"Subject: {subject_name}\n\n"
        f"Conversation:\n{block_text}"
    )
    _log.info(
        "SUMMARIZE subject=%r %d msg(s) -> meta_model=%s",
        subject_name, len(messages), cfg.get("meta_model", "?"),
    )
    try:
        result = _call_llm(_SUMMARIZE_SYSTEM_PROMPT, user_prompt, cfg, model_key="meta_model")
        _log.info("SUMMARIZE result: %d chars", len(result))
        return result
    except (httpx.HTTPError, KeyError, ValueError) as e:
        _log.warning("SUMMARIZE failed (%s) -> fallback returned", type(e).__name__)
        return f"(summary unavailable for subject '{subject_name}')"


def update_subject_memory(
    subject_name: str,
    existing_view: str,
    new_messages: list[dict],
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """
    Produce an updated materialized view, a log entry, and an epistemic status
    for a subject by merging new messages into the existing view.

    Returns: (view, log_entry, status)
    - view: the new materialized view content (markdown)
    - log_entry: brief description of what changed
    - status: "established" | "extended" | "revised"

    Falls back to (existing_view, fallback_message, "established") on failure.
    """
    if cfg is None:
        cfg = _config.load()
    if not new_messages:
        return existing_view, "", "established" if not existing_view else "extended"

    msgs_text = "\n\n".join(
        f"[{m['role'].upper()} relevance={float(m.get('relevance', 1.0)):.2f}]: {m['content']}"
        for m in new_messages
    )
    sections: list[str] = []
    if existing_view:
        sections.append(f"## Current view\n\n{existing_view}")
    sections.append(f"## New messages for: {subject_name}\n\n{msgs_text}")
    user_prompt = "\n\n".join(sections)

    _log.info(
        "MEMORY WRITE subject=%r existing=%d chars new=%d msg(s) -> subagent_model=%s",
        subject_name, len(existing_view), len(new_messages), cfg.get("subagent_model", "?"),
    )
    t0 = time.monotonic()
    try:
        raw = _call_llm(_UPDATE_MEMORY_SYSTEM_PROMPT, user_prompt, cfg, model_key="subagent_model")
        # Strip markdown fencing if the model wrapped the response despite instructions.
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        view = str(parsed.get("view", existing_view)).strip()
        log_entry = str(parsed.get("log_entry", "")).strip()
        status = str(parsed.get("status", "")).strip()
        if status not in {"established", "extended", "revised"}:
            status = "established" if not existing_view else "extended"
        elapsed = time.monotonic() - t0
        _log.info(
            "MEMORY WRITE result: subject=%r view=%d chars status=%s elapsed=%.1fs",
            subject_name, len(view), status, elapsed,
        )
        return view, log_entry, status
    except (json.JSONDecodeError, ValueError, httpx.HTTPError, KeyError) as e:
        elapsed = time.monotonic() - t0
        _log.warning(
            "MEMORY WRITE failed for subject=%r (%s) elapsed=%.1fs -> keeping existing view",
            subject_name, type(e).__name__, elapsed,
        )
        return existing_view, f"(memory update failed: {type(e).__name__})", "established"


def finalize_subject_memory(
    subject_name: str,
    existing_view: str,
    ranked_messages: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Produce the definitive final materialized view for an archived subject by
    synthesizing its ranked source messages into one coherent, non-redundant summary.

    ranked_messages is a list of dicts (keys: speaker, raw_content, timestamp_utc,
    confidence) ordered by confidence DESC then timestamp_utc DESC. Messages are
    appended to the prompt until finalize_max_chars is reached, so the model
    always receives a bounded payload regardless of subject history size.

    This is a one-shot synthesis pass, not an incremental update. It runs
    asynchronously after close_subject has completed Phase 1 compaction.

    Returns: (view, log_entry)
    - view: the final canonical view to write back to the memory file
    - log_entry: description of the finalization pass for the event log

    Falls back to (existing_view, fallback_message) on any failure.
    """
    if cfg is None:
        cfg = _config.load()

    max_chars = int(cfg.get("finalize_max_chars", 6000))

    sections: list[str] = []
    if existing_view:
        sections.append(f"## Current materialized view\n\n{existing_view}")

    included: list[str] = []
    chars_used = 0
    for msg in ranked_messages:
        ts = str(msg.get("timestamp_utc") or "")[:10]
        speaker = str(msg.get("speaker") or "").strip() or "Unknown"
        content = str(msg.get("raw_content") or "").strip()
        conf = float(msg.get("confidence") or 1.0)
        line = f"[{ts} | conf={conf:.2f}] {speaker}: {content}"
        if chars_used + len(line) > max_chars:
            break
        included.append(line)
        chars_used += len(line)

    if included:
        sections.append(
            f"## Source messages for: {subject_name}\n\n" + "\n\n".join(included)
        )

    if not sections:
        return existing_view, "(finalization skipped: no content to synthesize)"

    user_prompt = "\n\n".join(sections)
    _log.info(
        "FINALIZE subject=%r view=%d chars messages=%d/%d included -> meta_model=%s",
        subject_name, len(existing_view),
        len(included), len(ranked_messages),
        cfg.get("meta_model", "?"),
    )
    try:
        raw = _call_llm(_FINALIZE_MEMORY_SYSTEM_PROMPT, user_prompt, cfg, model_key="meta_model")
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        view = str(parsed.get("view", existing_view)).strip()
        log_entry = str(parsed.get("log_entry", "")).strip()
        _log.info(
            "FINALIZE result: subject=%r view=%d chars",
            subject_name, len(view),
        )
        return view, log_entry
    except (json.JSONDecodeError, ValueError, httpx.HTTPError, KeyError) as e:
        _log.warning(
            "FINALIZE failed for subject=%r (%s) -> keeping existing view",
            subject_name, type(e).__name__,
        )
        return existing_view, f"(finalization failed: {type(e).__name__})"


def split_subject_memory(
    subject_name: str,
    existing_view: str,
    event_log: str,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Ask the meta-model to partition an oversized subject into 2-5 child subjects.

    Returns a list of dicts, each with 'name' and 'view' keys. The list has at
    least 2 entries and at most 5. Returns an empty list on any failure so the
    caller can skip the split rather than crash.
    """
    if cfg is None:
        cfg = _config.load()

    sections: list[str] = [f"## Parent subject: {subject_name}"]
    if existing_view:
        sections.append(f"## Current materialized view\n\n{existing_view}")
    if event_log:
        sections.append(f"## Full event log\n\n{event_log}")

    user_prompt = "\n\n".join(sections)
    _log.info(
        "SPLIT subject=%r view=%d chars log=%d chars -> meta_model=%s",
        subject_name, len(existing_view), len(event_log), cfg.get("meta_model", "?"),
    )
    try:
        raw = _call_llm(_SPLIT_SUBJECT_SYSTEM_PROMPT, user_prompt, cfg, model_key="meta_model")
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        children = parsed.get("children", [])
        if not isinstance(children, list):
            raise ValueError("'children' is not a list")
        # Enforce 2-5 constraint
        children = [c for c in children if isinstance(c, dict) and c.get("name") and c.get("view")]
        if len(children) < 2:
            raise ValueError(f"model returned {len(children)} valid children (need >= 2)")
        children = children[:5]
        _log.info(
            "SPLIT result: subject=%r -> %d children: %s",
            subject_name, len(children), [c["name"] for c in children],
        )
        return [{"name": str(c["name"]).strip(), "view": str(c["view"]).strip()} for c in children]
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        _log.warning(
            "SPLIT failed for subject=%r (%s) -> skipping split",
            subject_name, type(e).__name__,
        )
        return []
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "SPLIT unexpected error for subject=%r (%s) -> skipping split",
            subject_name, type(e).__name__,
        )
        return []


def identify_dross(
    messages: list[dict],
    cfg: dict[str, Any] | None = None,
) -> list[int]:
    """
    Return the 0-based indices of zero-information messages.
    Returns [] on any failure.
    """
    if cfg is None:
        cfg = _config.load()
    if not messages:
        return []

    msgs_text = json.dumps(
        [{"index": i, "role": m["role"], "content": m["content"][:300]}
         for i, m in enumerate(messages)],
        ensure_ascii=False,
    )
    _log.info(
        "DROSS %d msg(s) -> subagent_model=%s",
        len(messages), cfg.get("subagent_model", "?"),
    )
    try:
        raw = _call_llm(_DROSS_SYSTEM_PROMPT, msgs_text, cfg, model_key="subagent_model")
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        result = [int(i) for i in parsed if isinstance(i, (int, str))]
        _log.info("DROSS result: %d of %d flagged as zero-info", len(result), len(messages))
        return result
    except (json.JSONDecodeError, ValueError, httpx.HTTPError, KeyError) as e:
        _log.warning("DROSS failed (%s) -> none flagged", type(e).__name__)
        return []


def summarize_dropped_content(
    messages: list[dict],
    cfg: dict[str, Any] | None = None,
) -> str:
    """
    Produce a brief orientation summary of messages that were dropped by context
    truncation. Used for truncation recovery injection.
    Returns the summary string.
    """
    if cfg is None:
        cfg = _config.load()
    return generate_subject_summary(messages, subject_name="(dropped context)", cfg=cfg)
