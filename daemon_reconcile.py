"""
daemon_reconcile.py - HuBrIS relation detection daemon.

Single job: drain reconcile_link actions from the memory_actions queue.

  reconcile_link - for a newly written semantic_link, embed its message,
                   search for semantically similar neighbors within the same
                   subject (using semantic_search_subject_messages), and ask
                   the LLM to classify each candidate pair as:
                     supports   - new memory corroborates the older one
                     contradicts - genuine conflict; both may still be valid
                     updates    - older fact is superseded by the new one
                   Applies an additive confidence adjustment to the older
                   link for each verdict, then stamps reconciled_at on the
                   new link to prevent re-processing.
                   Payload shape:
                     {"link_id": int, "message_id": str,
                      "subject_id": str, "created_date_utc": str}

Also exports _WATCH_SPEC so daemon_watcher discovers this daemon's scanner at
startup and enqueues eligible links on a schedule without daemon_watcher needing
any hardcoded knowledge of this daemon.

Run with: python daemon_reconcile.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config as _config
import db as _db
import embeddings as _emb
from backend_adapters import build_backend_adapter
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.reconcile")

ACTOR_RECONCILE = "MemoryReconciler"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_RECONCILE_SYSTEM_PROMPT = """\
You are HuBrIS, a memory reconciliation agent. Your job is to compare a newly \
recorded memory to a set of older memories on the same subject and classify the \
relationship of each older memory to the new one.

Return a JSON array. Each element must have exactly two keys:
  "link_id" : the integer id of the OLDER memory
  "relation" : one of "supports", "contradicts", or "updates"

Definitions:
  supports    - the new memory is consistent with and corroborates the older one
  contradicts - genuine conflict; both facts may still be valid but disagree
  updates     - the older fact is superseded or deprecated by the new one

Omit neutral pairs (no significant relationship). Return [] if all pairs are neutral.
Return only valid JSON. No explanation, no preamble.\
"""


# ---------------------------------------------------------------------------
# Watch spec scanner (called by daemon_watcher on a schedule)
# ---------------------------------------------------------------------------


def _scan_pending_reconciliations(root: Path) -> list[dict]:
    """
    Return items ready to be enqueued as reconcile_link actions.

    Each item shape:
      subject_id - str(link_id), used as the dedup key in memory_actions
      payload    - all data the handler needs

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    Returns empty list when sqlite-vec is not installed (embeddings unavailable),
    because the handler cannot process reconcile_link actions without them.
    """
    if not _emb.is_available():
        return []
    try:
        links = _db.get_links_pending_reconciliation(root)
    except Exception as exc:
        _log.warning("RECONCILE scan: DB query failed: %s", exc)
        return []

    return [
        {
            "subject_id": str(link["id"]),
            "payload": {
                "link_id": link["id"],
                "message_id": link["message_id"],
                "subject_id": link["subject_id"],
                "created_date_utc": link["created_date_utc"],
            },
        }
        for link in links
    ]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_reconcile_link(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Reconcile one semantic_link against older memories in the same subject.

    Steps:
    1. Fetch the message text.
    2. Embed it.
    3. Search for neighbors scoped to the same subject.
    4. Filter to candidates older than this link.
    5. If no candidates exist, mark reconciled and return.
    6. Ask the LLM to classify each candidate pair.
    7. Apply additive confidence adjustments to older links.
    8. Stamp reconciled_at on this link.
    """
    link_id: int = int(payload["link_id"])
    message_id: str = str(payload["message_id"])
    link_subject_id: str = str(payload["subject_id"])
    created_date_utc: str = str(payload.get("created_date_utc", ""))

    # --- Step 1: fetch message text ---
    message = _db.get_message_by_id(message_id, root)
    if message is None:
        _log.warning("RECONCILE: link_id=%d message_id=%r not found; marking reconciled", link_id, message_id)
        _db.mark_link_reconciled(link_id, ACTOR_RECONCILE, root)
        return

    raw_content: str = str(message.get("raw_content", "") or "")
    if not raw_content.strip():
        _log.info("RECONCILE: link_id=%d empty message; marking reconciled", link_id)
        _db.mark_link_reconciled(link_id, ACTOR_RECONCILE, root)
        return

    # --- Step 2: embed ---
    if not _emb.is_available():
        _log.debug("RECONCILE: embeddings not available; skipping link_id=%d", link_id)
        return

    embedding_bytes = _emb.embed(raw_content, cfg)
    if embedding_bytes is None:
        _log.warning("RECONCILE: embed failed for link_id=%d; will retry", link_id)
        raise RuntimeError("embed returned None")

    # --- Step 3: subject-scoped neighbor search ---
    k: int = int(cfg.get("reconcile_candidates_k", 5))
    candidates = _db.semantic_search_subject_messages(embedding_bytes, link_subject_id, k, root)

    # --- Step 4: keep only neighbors with older created_date_utc ---
    older = [
        c for c in candidates
        if c.get("link_id") != link_id
        and str(c.get("link_created_date", "") or "") < created_date_utc
    ]

    if not older:
        _log.info(
            "RECONCILE: link_id=%d subject=%r no older neighbors; marking reconciled",
            link_id, link_subject_id,
        )
        _db.mark_link_reconciled(link_id, ACTOR_RECONCILE, root)
        return

    # --- Step 5: build judgment prompt ---
    speaker = str(message.get("speaker", ""))
    new_memory_text = f"[{speaker}]: {raw_content}" if speaker else raw_content

    candidate_lines = []
    for cand in older:
        cand_speaker = str(cand.get("speaker", ""))
        cand_content = str(cand.get("raw_content", "") or "")
        cand_link_id = int(cand.get("link_id", 0))
        cand_date = str(cand.get("link_created_date", "") or "")
        label = f"[{cand_speaker}]: {cand_content}" if cand_speaker else cand_content
        candidate_lines.append(f'  {{"id": {cand_link_id}, "recorded_at": "{cand_date}", "content": {json.dumps(label)}}}')

    user_prompt = (
        "New memory:\n"
        f"  {json.dumps(new_memory_text)}\n\n"
        "Older memories in same subject:\n"
        + "\n".join(candidate_lines)
        + "\n\nFor each older memory, classify its relationship to the new memory."
    )

    # --- Step 6: LLM judgment pass ---
    model = cfg.get("subagent_model", cfg.get("meta_model", ""))
    if not model:
        _log.warning("RECONCILE: no subagent_model configured; marking reconciled without judgment")
        _db.mark_link_reconciled(link_id, ACTOR_RECONCILE, root)
        return

    try:
        raw_response = build_backend_adapter(cfg).complete(
            _RECONCILE_SYSTEM_PROMPT, user_prompt, model
        )
    except Exception as exc:
        _log.warning("RECONCILE: LLM call failed for link_id=%d: %s; will retry", link_id, exc)
        raise

    # --- Step 7: parse and apply adjustments ---
    verdicts = _parse_verdicts(raw_response)

    delta_supports = float(cfg.get("reconcile_delta_supports", 0.05))
    delta_contradicts = float(cfg.get("reconcile_delta_contradicts", -0.10))
    delta_updates = float(cfg.get("reconcile_delta_updates", -0.20))

    # Build a confidence lookup from the candidates we already have
    confidence_by_link_id: dict[int, float] = {
        int(c["link_id"]): float(c.get("link_confidence", 1.0))
        for c in older
        if c.get("link_id") is not None
    }
    valid_older_ids = {int(c["link_id"]) for c in older if c.get("link_id") is not None}

    for verdict in verdicts:
        old_link_id = verdict.get("link_id")
        relation = verdict.get("relation", "")
        if not isinstance(old_link_id, int) or old_link_id not in valid_older_ids:
            _log.debug("RECONCILE: verdict for unknown link_id=%r; skipping", old_link_id)
            continue

        delta = {
            "supports": delta_supports,
            "contradicts": delta_contradicts,
            "updates": delta_updates,
        }.get(relation)

        if delta is None:
            _log.debug("RECONCILE: unknown relation %r for link_id=%d; skipping", relation, old_link_id)
            continue

        current = confidence_by_link_id.get(old_link_id, 1.0)
        new_confidence = max(0.0, min(1.0, current + delta))
        _db.attenuate_link_confidence(old_link_id, new_confidence, ACTOR_RECONCILE, root)
        _db.insert_subject_relation(
            from_link_id=old_link_id,
            to_link_id=link_id,
            subject_id=link_subject_id,
            relation=relation,
            actor=ACTOR_RECONCILE,
            root=root,
        )
        _log.info(
            "RECONCILE: link_id=%d -> old_link_id=%d relation=%r %.4f -> %.4f",
            link_id, old_link_id, relation, current, new_confidence,
        )

    # --- Step 8: mark this link reconciled ---
    _db.mark_link_reconciled(link_id, ACTOR_RECONCILE, root)
    _log.info("RECONCILE: link_id=%d reconciled (%d verdicts applied)", link_id, len(verdicts))


def _parse_verdicts(raw: str) -> list[dict[str, Any]]:
    """
    Parse the LLM JSON response into a list of verdict dicts.
    Returns [] on any parse failure rather than raising.
    """
    if not raw or not raw.strip():
        return []
    # Strip markdown code fences if the model wraps its output
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as exc:
        _log.warning("RECONCILE: could not parse LLM response as JSON: %s | raw=%r", exc, raw[:200])
        return []
    if not isinstance(parsed, list):
        _log.warning("RECONCILE: expected JSON array, got %r", type(parsed).__name__)
        return []
    return [v for v in parsed if isinstance(v, dict)]


# ---------------------------------------------------------------------------
# Watch spec (read by daemon_watcher at startup via _WATCH_SPEC discovery)
# ---------------------------------------------------------------------------

_WATCH_SPEC: list[dict[str, Any]] = [
    {
        "interval_s": _config.DEFAULTS.get("reconcile_interval_s", 60),
        "action_type": "reconcile_link",
        "actor": ACTOR_RECONCILE,
        "scanner": _scan_pending_reconciliations,
    }
]


# ---------------------------------------------------------------------------
# Action handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "reconcile_link": _handle_reconcile_link,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="reconcile",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=ACTOR_RECONCILE,
    )
    daemon.run()


if __name__ == "__main__":
    main()
