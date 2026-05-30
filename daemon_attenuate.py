"""
daemon_attenuate.py - HuBrIS confidence attenuation daemon.

Single job: drain attenuate_confidence actions from the memory_actions queue.

  attenuate_confidence - multiply a semantic_link's confidence by
                         _ATTENUATION_FACTOR and write it back.
                         Payload shape:
                           {"link_id": int, "message_id": str,
                            "subject_id": str, "current_confidence": float}

Also exports _WATCH_SPEC so daemon_watcher discovers this daemon's scanner at
startup and enqueues eligible links on a schedule without daemon_watcher needing
any hardcoded knowledge of this daemon.

Run with: python daemon_attenuate.py
"""

from pathlib import Path
from typing import Any

import config as _config
import db as _db
from daemon_base import DaemonBase
from log import get_logger

_log = get_logger("hubris.attenuate")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATTENUATION_FACTOR: float = 0.975
_ELIGIBILITY_DAYS: int = 7


# ---------------------------------------------------------------------------
# Watch spec scanner (called by daemon_watcher on a schedule)
# ---------------------------------------------------------------------------


def _scan_eligible_links(root: Path) -> list[dict]:
    """
    Return items ready to be enqueued as attenuate_confidence actions.

    Each item shape:
      subject_id - str(link_id), used as the dedup key in memory_actions
      payload    - all data the handler needs to apply attenuation

    Called by daemon_watcher._run_watch_spec_thread; must have no side effects.
    """
    try:
        links = _db.get_links_eligible_for_attenuation(_ELIGIBILITY_DAYS, root)
    except Exception as exc:
        _log.warning("ATTENUATE scan: DB query failed: %s", exc)
        return []

    return [
        {
            "subject_id": str(link["id"]),
            "payload": {
                "link_id": link["id"],
                "message_id": link["message_id"],
                "subject_id": link["subject_id"],
                "current_confidence": link["confidence"],
            },
        }
        for link in links
    ]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _handle_attenuate_confidence(
    root: Path,
    cfg: dict,
    subject_id: str | None,
    payload: dict,
) -> None:
    """
    Apply one attenuation step to a semantic_link.

    Multiplies the link's confidence by _ATTENUATION_FACTOR and writes the
    result back. Confidence decays exponentially toward zero and is never
    deleted or clamped.
    """
    link_id: int = int(payload["link_id"])
    current: float = float(payload["current_confidence"])
    new_confidence: float = current * _ATTENUATION_FACTOR

    _db.attenuate_link_confidence(link_id, new_confidence, _db.ACTOR_CONFIDENCE_ATTENUATOR, root)
    _log.info(
        "ATTENUATE: link_id=%d %.4f -> %.4f",
        link_id, current, new_confidence,
    )


# ---------------------------------------------------------------------------
# Watch spec (read by daemon_watcher at startup via _WATCH_SPEC discovery)
# ---------------------------------------------------------------------------

_WATCH_SPEC: list[dict[str, Any]] = [
    {
        "interval_s": 3600,
        "action_type": "attenuate_confidence",
        "actor": _db.ACTOR_CONFIDENCE_ATTENUATOR,
        "scanner": _scan_eligible_links,
    }
]


# ---------------------------------------------------------------------------
# Action handler registry
# ---------------------------------------------------------------------------

_MEMORY_ACTION_HANDLERS: dict[str, Any] = {
    "attenuate_confidence": _handle_attenuate_confidence,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _config.load()
    root = _config.memory_root(cfg.get("workspace_id", "global"))

    daemon = DaemonBase(
        name="attenuate",
        root=root,
        action_handlers=_MEMORY_ACTION_HANDLERS,
        actor=_db.ACTOR_CONFIDENCE_ATTENUATOR,
    )
    daemon.run()


if __name__ == "__main__":
    main()
