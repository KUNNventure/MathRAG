"""Shared UI helpers for Ingestion / Query trace pages."""

from __future__ import annotations

from typing import Any, List, Tuple

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

TRACE_PAGE_SIZE = 20
UI_TEXT_CAP = 2000
UI_JSON_TEXT_CAP = 4000


def paginate_traces(
    traces: List[Any],
    *,
    page_key: str = "trace_list_page",
) -> Tuple[List[Any], int, int]:
    """Return (page_slice, page_index_0based, total_pages)."""
    if not traces:
        return [], 0, 1
    total_pages = max(1, (len(traces) + TRACE_PAGE_SIZE - 1) // TRACE_PAGE_SIZE)
    if len(traces) <= TRACE_PAGE_SIZE:
        return traces, 0, 1
    page_1 = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=page_key,
    )
    page_idx = int(page_1) - 1
    start = page_idx * TRACE_PAGE_SIZE
    end = start + TRACE_PAGE_SIZE
    st.caption(f"Showing {start + 1}–{min(end, len(traces))} of {len(traces)} traces")
    return traces[start:end], page_idx, total_pages


def truncate_text(text: str, cap: int = UI_TEXT_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + "\n\n… [truncated]"


def render_truncated_text_area(
    text: str,
    *,
    key: str,
    cap: int = UI_TEXT_CAP,
    max_height: int = 400,
) -> None:
    """Show text in a disabled text_area; long content is truncated."""
    shown = truncate_text(text or "", cap=cap)
    height = max(80, min(len(shown) // 2, max_height))
    st.text_area(
        key,
        value=shown,
        height=height,
        disabled=True,
        label_visibility="collapsed",
    )
