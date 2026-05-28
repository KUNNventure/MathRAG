"""TraceService – read and parse traces from logs/traces.jsonl.

Provides a typed, filterable interface over the raw JSONL trace log.
Uses streaming reads + a bounded heap so large files do not load every trace
into memory at once.
"""

from __future__ import annotations

import heapq
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")

# Skip absurdly long lines (corrupt / non-JSON) to bound memory per line
MAX_LINE_BYTES = 50 * 1024 * 1024


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order by ``started_at``.

        Scans the JSONL file once, keeping at most ``limit`` matches in a
        bounded heap (memory ~O(limit)), so very large trace logs do not
        require loading all entries.
        """
        cap = max(1, int(limit))
        heap: list[tuple[str, Dict[str, Any]]] = []

        for t in self._iter_trace_dicts():
            if trace_type and t.get("trace_type") != trace_type:
                continue
            sa = str(t.get("started_at") or "")
            if len(heap) < cap:
                heapq.heappush(heap, (sa, t))
            else:
                if sa > heap[0][0]:
                    heapq.heapreplace(heap, (sa, t))

        ordered = sorted(heap, key=lambda x: x[0], reverse=True)
        return [tr for _, tr in ordered]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``.

        Returns:
            Trace dict, or ``None`` if not found.
        """
        for t in self._iter_trace_dicts():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace.

        Returns:
            List of dicts with keys: stage_name, elapsed_ms, data.
            Ordered by appearance.
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_trace_dicts(self) -> Iterator[Dict[str, Any]]:
        """Yield one trace dict per valid JSON line (streaming)."""
        if not self.traces_path.exists():
            return
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                if len(raw) > MAX_LINE_BYTES:
                    logger.warning(
                        "Skipping oversize trace line (%d bytes) in %s",
                        len(raw),
                        self.traces_path,
                    )
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", raw[:80])
                    continue
                if isinstance(obj, dict):
                    yield obj
