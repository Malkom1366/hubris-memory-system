"""
Logging setup for HuBrIS.

Architecture:
  Each process gets one in-memory queue and one QueueListener drain thread.
  Callers use QueueHandler, which posts records without touching the disk.
  The listener thread drains the queue and writes to the rotating file in
  batches, so the hot path (daemon code) never blocks on I/O or file locks.

  On Windows, multiple processes can race on log rotation; the
  _SafeRotatingFileHandler falls back to truncation instead of silently
  dropping writes when the rename fails.

  File: ~/.hubris/logs/hubris.log
  Max size: 2 MB. Keeps up to 20 backups (hubris.log.1 ... hubris.log.20).

Usage:
    from log import get_logger
    _log = get_logger("hubris.server")
"""

import atexit
import logging
import logging.handlers
import os
import queue
from pathlib import Path

_LOG_FILENAME = "hubris.log"
_MAX_BYTES = 2 * 1024 * 1024   # 2 MB per file
_BACKUP_COUNT = 20               # hubris.log.1 ... hubris.log.20
_FMT = "%(asctime)s.%(msecs)03d  %(levelname)-8s  [%(name)s]  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialized: set[str] = set()

# Per-process in-memory record queue.  Unbounded so enqueue never blocks.
_log_queue: queue.Queue = queue.Queue(-1)
_listener: logging.handlers.QueueListener | None = None


class _SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that falls back to truncation when rename fails.

    On Windows, doRollover() renames the current log file to hubris.log.1.
    That rename fails with PermissionError if another daemon process already
    has the file open.  The base class wraps the entire emit() in a try/except
    and silently drops the write when doRollover() raises, leaving all daemons
    completely silent.

    Override the rotator hook so that when rename is impossible we truncate
    the file in place instead.  We lose the rolled-over content, but writes
    resume immediately and logging is never silently dropped.
    """

    @staticmethod
    def _safe_rotate(source: str, dest: str) -> None:
        try:
            if os.path.exists(dest):
                os.remove(dest)
            os.rename(source, dest)
        except OSError:
            # File locked by a sibling daemon process on Windows.
            # Truncate in place so the next open() starts from byte 0.
            try:
                open(source, "w").close()  # noqa: WPS515
            except OSError:
                pass

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rotator = self._safe_rotate


def _ensure_listener() -> None:
    """Start the per-process drain thread the first time any logger is created."""
    global _listener
    if _listener is not None:
        return

    log_dir = Path.home() / ".hubris" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _LOG_FILENAME

    file_handler = _SafeRotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    # respect_handler_level ensures DEBUG/INFO/WARNING filters on the file
    # handler are honoured even though QueueHandler passes everything through.
    _listener = logging.handlers.QueueListener(
        _log_queue, file_handler, respect_handler_level=True
    )
    _listener.start()
    # Flush remaining records on clean process exit.
    atexit.register(_listener.stop)


def get_logger(name: str = "hubris") -> logging.Logger:
    """Return a named logger. Safe to call multiple times with the same name."""
    logger = logging.getLogger(name)
    if name in _initialized:
        return logger

    _ensure_listener()

    # QueueHandler never touches the file - it just enqueues the LogRecord.
    # The drain thread (QueueListener) is the only writer to the file.
    handler = logging.handlers.QueueHandler(_log_queue)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   # don't bubble up to root logger

    _initialized.add(name)
    return logger


def attach_external_logger(name: str, level: int = logging.WARNING) -> None:
    """Route an external library's logger (e.g. 'mcp') to hubris.log.

    Call this once at startup to capture FastMCP tool-exception tracebacks
    that would otherwise be lost because the 'mcp.*' loggers have no handlers
    and do not propagate to the root logger.

    Safe to call multiple times - duplicate QueueHandlers are not added.
    """
    _ensure_listener()
    logger = logging.getLogger(name)
    for h in logger.handlers:
        if isinstance(h, logging.handlers.QueueHandler) and h.queue is _log_queue:
            return  # already attached
    handler = logging.handlers.QueueHandler(_log_queue)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
