"""Query Traces page – browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
4. Per-stage detail (read-only)
5. Optional Ragas single-trace evaluation (API cost, user-confirmed)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.pages._trace_ui import (
    TRACE_PAGE_SIZE,
    paginate_traces,
    render_truncated_text_area,
)
from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


@st.cache_data(ttl=30, show_spinner=False)
def _load_traces_cached():
    """Cached trace loader – avoids re-parsing traces.jsonl every render."""
    svc = TraceService()
    return svc.list_traces(trace_type="query")


def render() -> None:
    """Render the Query Traces page."""
    st.header("🔎 Query Traces")
    st.caption(
        "只读查看各阶段与检索结果。每条 trace 底部可 **运行 Ragas 单条评估**（需勾选费用确认，会调评测 API）。"
    )

    svc = TraceService()
    traces = _load_traces_cached()

    if not traces:
        st.info("No query traces recorded yet. Run a query first!")
        return

    # ── Keyword filter ─────────────────────────────────────────────
    keyword = st.text_input(
        "Search by query keyword",
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"📋 Query History ({len(traces)})")
    page_traces, page_base, _ = paginate_traces(traces, page_key="query_trace_page")

    for local_idx, trace in enumerate(page_traces):
        idx = page_base * TRACE_PAGE_SIZE + local_idx
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── Expander title: show query text ────────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        expander_title = (
            f"🔍 \"{query_preview}\"  ·  {total_label}  ·  {started[:19]}"
        )

        with st.expander(expander_title, expanded=False):
            # ── 1. Query overview ──────────────────────────────
            st.markdown("#### 💬 Query")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                source_emoji = "🤖" if source == "mcp" else "📡"
                st.markdown(f"**Source:** {source_emoji} `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '—')}`")
                st.markdown(f"**Collection:** `{meta.get('collection', '—')}`")

            st.divider()

            # ── 2. Overview metrics ────────────────────────────
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = (stages_by_name.get("dense_retrieval", {}).get("data") or {})
            sparse_d = (stages_by_name.get("sparse_retrieval", {}).get("data") or {})
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            with rc1:
                st.metric("Dense Hits", dense_count)
            with rc2:
                st.metric("Sparse Hits", sparse_count)
            with rc3:
                st.metric("Fused", fusion_count or (dense_count + sparse_count))
            with rc4:
                st.metric("After Rerank", rerank_count if rerank_d else "—")
            with rc5:
                st.metric("Total Time", total_label)

            # ── Diagnostic hints ───────────────────────────────
            _render_diagnostics(
                stages_by_name, dense_d, sparse_d, fusion_d, rerank_d,
                dense_count, sparse_count,
            )

            st.divider()

            # ── 3. Stage timing waterfall ──────────────────────
            main_stage_names = ("query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank")
            main_timings = [t for t in timings if t["stage_name"] in main_stage_names]
            if main_timings:
                st.markdown("#### ⏱️ Stage Timings")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_timings}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "Stage": t["stage_name"],
                        "Elapsed (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_timings
                ])

            st.divider()

            # ── 4. Per-stage detail tabs ───────────────────────
            st.markdown("#### 🔍 Stage Details")

            tab_defs = []
            if "query_processing" in stages_by_name:
                tab_defs.append(("🔤 Query Processing", "query_processing"))
            if "dense_retrieval" in stages_by_name:
                tab_defs.append(("🟦 Dense Retrieval", "dense_retrieval"))
            if "sparse_retrieval" in stages_by_name:
                tab_defs.append(("🟨 Sparse Retrieval", "sparse_retrieval"))
            if "fusion" in stages_by_name:
                tab_defs.append(("🟩 Fusion (RRF)", "fusion"))
            if "rerank" in stages_by_name:
                tab_defs.append(("🟪 Rerank", "rerank"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "query_processing":
                            _render_query_processing_stage(data)
                        elif key == "dense_retrieval":
                            _render_retrieval_stage(data, "Dense", trace_idx=idx)
                        elif key == "sparse_retrieval":
                            _render_retrieval_stage(data, "Sparse", trace_idx=idx)
                        elif key == "fusion":
                            _render_fusion_stage(data, trace_idx=idx)
                        elif key == "rerank":
                            _render_rerank_stage(data, trace_idx=idx)
            else:
                st.info("No stage details available.")

            _render_trace_ragas_eval_section(stages_by_name, meta, idx)


def _render_trace_ragas_eval_section(
    stages_by_name: Dict[str, Any],
    meta: Dict[str, Any],
    idx: int,
) -> None:
    """Optional Ragas LLM-as-Judge on chunks embedded in this trace."""
    from src.observability.dashboard.services.trace_ragas import (
        fallback_answer_from_chunks,
        pick_trace_chunks_for_eval,
        run_ragas_for_trace,
    )

    st.divider()
    st.markdown("#### 📏 Ragas 单条评估")
    st.caption(
        "使用 trace 内检索片段作为 Context。faithfulness / context_precision 等需要「回答」文本；"
        "可粘贴当时模型输出，或启用占位拼接（指标仅供参考）。"
    )

    raw_chunks = pick_trace_chunks_for_eval(stages_by_name)
    q = str(meta.get("query") or meta.get("effective_query_preview") or "").strip()

    if not raw_chunks:
        st.warning("未找到带文本的 chunks（需存在 rerank / fusion / dense / sparse 任一阶段的 chunks）。")
        return
    if not q:
        st.warning("本条 trace 缺少 query 字段，无法评估。")
        return

    st.text_area(
        "模型回答（Answer）",
        height=100,
        placeholder="粘贴该次查询的系统回答；可留空并在下方启用占位答案。",
        key=f"qt_ragas_ans_{idx}",
    )
    st.checkbox(
        "回答为空时用检索片段拼接占位答案",
        value=True,
        key=f"qt_ragas_fb_{idx}",
    )
    st.text_input(
        "Judge 模型覆盖（可选）",
        value="",
        key=f"qt_ragas_judge_{idx}",
        help="留空则使用 settings 中的默认 LLM。",
    )
    st.checkbox("我已知晓将产生评测 API 费用", value=False, key=f"qt_ragas_cost_{idx}")

    b1, b2 = st.columns(2)
    with b1:
        run = st.button("运行 Ragas 评估", type="primary", key=f"qt_ragas_run_{idx}")
    with b2:
        if st.button("清除本条评估结果", key=f"qt_ragas_clr_{idx}"):
            st.session_state.pop(f"qt_ragas_res_{idx}", None)
            st.rerun()

    res_key = f"qt_ragas_res_{idx}"
    if run:
        if not st.session_state.get(f"qt_ragas_cost_{idx}"):
            st.warning("请先勾选费用确认。")
        elif not q:
            pass
        else:
            ans = (st.session_state.get(f"qt_ragas_ans_{idx}") or "").strip()
            if not ans and st.session_state.get(f"qt_ragas_fb_{idx}", True):
                ans = fallback_answer_from_chunks(raw_chunks)
            if not ans:
                st.warning("请填写回答，或启用占位答案。")
            else:
                judge_raw = (st.session_state.get(f"qt_ragas_judge_{idx}") or "").strip()
                judge_model = judge_raw or None
                with st.spinner("Ragas 评估中…"):
                    try:
                        scores = run_ragas_for_trace(q, raw_chunks, ans, judge_model=judge_model)
                    except Exception as exc:
                        st.error(f"评估失败: {exc}")
                        logger.exception("Trace Ragas eval failed")
                    else:
                        st.session_state[res_key] = scores
                        st.success("评估完成。")
                        st.rerun()

    saved = st.session_state.get(res_key)
    if isinstance(saved, dict) and saved:
        st.markdown("**最近一次指标**")
        cols = st.columns(min(len(saved), 4))
        for i, (name, val) in enumerate(sorted(saved.items())):
            with cols[i % len(cols)]:
                st.metric(name.replace("_", " ").title(), f"{float(val):.4f}")


def _render_diagnostics(
    stages_by_name: Dict[str, Any],
    dense_d: Dict[str, Any],
    sparse_d: Dict[str, Any],
    fusion_d: Dict[str, Any],
    rerank_d: Dict[str, Any],
    dense_count: int,
    sparse_count: int,
) -> None:
    """Render diagnostic hints about missing or errored pipeline stages."""
    hints: list = []

    # Dense errors
    dense_err = dense_d.get("error", "")
    if dense_err:
        hints.append(("error", f"**Dense Retrieval failed:** {dense_err}"))
    elif dense_count == 0 and "dense_retrieval" in stages_by_name:
        hints.append(("warning", "Dense Retrieval returned **0 results**. Check if the collection has indexed data."))

    # Sparse errors / empty
    sparse_err = sparse_d.get("error", "")
    if sparse_err:
        hints.append(("error", f"**Sparse Retrieval failed:** {sparse_err}"))
    elif sparse_count == 0 and "sparse_retrieval" in stages_by_name:
        hints.append((
            "warning",
            "Sparse (BM25) Retrieval returned **0 results**. "
            "BM25 index may be empty or not yet built for this collection.",
        ))

    # Fusion missing
    if "fusion" not in stages_by_name:
        if dense_count > 0 and sparse_count > 0:
            hints.append(("info", "Fusion stage was not recorded even though both retrievers returned results."))
        elif dense_count == 0 or sparse_count == 0:
            only_source = "Dense" if dense_count > 0 else ("Sparse" if sparse_count > 0 else "neither")
            hints.append((
                "info",
                f"**Fusion (RRF) skipped:** only {only_source} retrieval returned results. "
                "Fusion requires both Dense and Sparse results to merge.",
            ))

    # Rerank missing
    if "rerank" not in stages_by_name:
        if dense_count > 0 or sparse_count > 0:
            hints.append((
                "info",
                "**Rerank skipped:** reranker is not enabled or not configured. "
                "Enable `reranker` in settings.yaml to apply LLM-based reranking.",
            ))

    # All results empty
    if dense_count == 0 and sparse_count == 0:
        hints.append((
            "warning",
            "**No results found.** The collection may be empty, or the query "
            "doesn't match any indexed content. Try ingesting data first.",
        ))

    # Render hints
    for level, msg in hints:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """Render Query Processing stage: original query → keywords."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Query**")
        st.info(data.get("original_query", "—"))
    with c2:
        st.markdown("**Method**")
        st.code(data.get("method", "—"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**Extracted Keywords**")
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("No keywords extracted.")


def _render_retrieval_stage(data: Dict[str, Any], label: str, *, trace_idx: int = 0) -> None:
    """Render Dense or Sparse retrieval stage: method, counts, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "—"))
        extra_label = "Provider" if "provider" in data else "Keywords"
        st.metric(extra_label, extra)
    with c3:
        st.metric("Results", data.get("result_count", 0))

    st.markdown(f"**Top-K requested:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk_{trace_idx}")
    else:
        st.info(f"No {label.lower()} results returned.")


def _render_fusion_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Fusion (RRF) stage: input lists, fused result count, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "rrf"))
    with c2:
        st.metric("Input Lists", data.get("input_lists", "—"))
    with c3:
        st.metric("Fused Results", data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"fusion_chunk_{trace_idx}")
    else:
        st.info("No fusion results.")


def _render_rerank_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Rerank stage: method, input/output counts, reranked chunk list."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        st.metric("Provider", data.get("provider", "—"))
    with c3:
        st.metric("Input", data.get("input_count", "—"))
    with c4:
        st.metric("Output", data.get("output_count", "—"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"rerank_chunk_{trace_idx}")
    else:
        st.info("No reranked results.")


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """Render a list of chunk dicts as a compact, readable table with expandable text."""
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # Colour-coded score indicator
        if score >= 0.8:
            score_bar = "🟢"
        elif score >= 0.5:
            score_bar = "🟡"
        else:
            score_bar = "🔴"

        header = f"{score_bar} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            if text:
                render_truncated_text_area(
                    text,
                    key=f"{prefix}_{ci}",
                    max_height=350,
                )
            else:
                st.caption("_No text available_")


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
