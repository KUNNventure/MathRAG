"""Observability logger utilities.

Provides:
- ``configure_logging``: one-time setup from settings (log level).
- ``get_logger``: standard human-readable logger.
- ``JSONFormatter``: custom :class:`logging.Formatter` that emits JSON.
- ``get_trace_logger``: returns a logger backed by a JSON Lines file handler.
- ``write_trace``: convenience function to append a trace dict to
  ``logs/traces.jsonl``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import resolve_path

# Default path for traces file (absolute, CWD-independent)
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")

_logging_configured = False
_active_log_level = "INFO"


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging once (idempotent).

    Args:
        log_level: Level name from settings (e.g. ``INFO``, ``DEBUG``).
    """
    global _logging_configured, _active_log_level
    normalized = (log_level or "INFO").upper()
    level = getattr(logging, normalized, logging.INFO)
    _active_log_level = normalized

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _logging_configured = True


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name.
        log_level: Optional override; re-runs :func:`configure_logging` when set.

    Returns:
        Configured logger instance.
    """
    if log_level:
        configure_logging(log_level)
    elif not _logging_configured:
        configure_logging(_active_log_level)
    return logging.getLogger(name)


# ── JSON Lines formatter ────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs one JSON object per line.

    Each log record is serialised to a dict containing at least:
    ``timestamp``, ``level``, ``logger``, ``message``.  If the record
    carries an ``exc_info`` tuple the traceback is included as
    ``exception``.

    Extra attributes attached via *extra=* on the logger call are
    merged into the top-level dict (except internal Python fields).
    """

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Return the log record as a single-line JSON string."""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # merge extra fields the caller attached
        for key, val in record.__dict__.items():
            if key not in self._INTERNAL_ATTRS and key not in payload:
                try:
                    json.dumps(val)  # cheap serialisability test
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── Trace logger ────────────────────────────────────────────────────


def get_trace_logger(
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    name: str = "modular-rag.trace",
    log_level: Optional[str] = None,
) -> logging.Logger:
    """Return a logger that writes JSON Lines to *traces_path*.

    The logger uses :class:`JSONFormatter` and a :class:`FileHandler`
    configured to append.  Repeated calls with the same *name* return
    the same logger (standard :mod:`logging` semantics).

    Args:
        traces_path: File path for the JSONL output.  Parent directories
            are created automatically.
        name: Logger name.
        log_level: Optional level override; defaults to last configured level.

    Returns:
        A :class:`logging.Logger` ready for JSON Lines output.
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    level_name = (log_level or _active_log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # don't echo to console

    return logger


# ── Convenience writer for trace dicts ──────────────────────────────


def write_trace(
    trace_dict: Dict[str, Any],
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
) -> None:
    """Append a single trace dictionary as one JSON line.

    This is a thin wrapper that writes directly — no logging
    framework involved — so the output is identical to what
    :class:`~src.core.trace.trace_collector.TraceCollector` produces.

    Args:
        trace_dict: A JSON-serialisable dictionary (typically from
            ``TraceContext.to_dict()``).
        traces_path: Output file path; parent directories are created
            automatically.
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(trace_dict, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
