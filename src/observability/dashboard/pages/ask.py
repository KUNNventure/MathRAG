"""Ask page - query the knowledge base directly from Dashboard."""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime
from dataclasses import replace as dc_replace
from typing import Any, Dict, List

import streamlit as st

from src.core.settings import load_settings, resolve_path
from src.core.trace import TraceCollector, TraceContext
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.response.answer_prompt import build_answer_messages
from src.libs.llm import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.query_engine.hybrid_search import HybridSearchResult
from src.observability.dashboard.services.data_service import DataService


def _ensure_ask_state() -> None:
    if "ask_round_history" not in st.session_state:
        st.session_state["ask_round_history"] = []
    if "ask_last_round" not in st.session_state:
        st.session_state["ask_last_round"] = None
    if "ask_show_chunks" not in st.session_state:
        st.session_state["ask_show_chunks"] = False


def _result_field(result: Any, field: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


def _result_metadata(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        metadata = result.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}
    metadata = getattr(result, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _build_components(
    settings: Any,
    collection: str,
    query_max_keywords: int,
    enable_filter_parsing: bool,
):
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.query_processor import QueryProcessor, QueryProcessorConfig
    from src.core.query_engine.reranker import create_core_reranker
    from src.core.query_engine.sparse_retriever import create_sparse_retriever

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)

    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection

    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(
            QueryProcessorConfig(
                max_keywords=query_max_keywords,
                enable_filter_parsing=enable_filter_parsing,
            )
        ),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    hybrid_search.fusion = RRFFusion(k=settings.retrieval.rrf_k)
    reranker = create_core_reranker(settings=settings)
    return hybrid_search, reranker


def _render_results(results: List[Any], key_prefix: str = "current") -> None:
    st.subheader(f"Results ({len(results)})")
    for idx, result in enumerate(results, start=1):
        metadata = _result_metadata(result)
        source_name = Path(metadata.get("source_path", "unknown")).name
        score = _result_field(result, "score", 0.0)
        title = metadata.get("title", "") or f"Chunk {idx}"
        chunk_id = _result_field(result, "chunk_id", "")
        with st.expander(f"#{idx} · {title} · score={score:.4f}", expanded=(idx == 1)):
            st.caption(f"source: {source_name}")
            if chunk_id:
                st.caption(f"id: `{chunk_id}`")
            st.text_area(
                "Content",
                value=_result_field(result, "text", "") or "",
                height=220,
                disabled=True,
                key=f"ask_chunk_{key_prefix}_{idx}_{chunk_id}",
                label_visibility="collapsed",
            )
            if metadata:
                with st.expander("Metadata", expanded=False):
                    st.json(metadata)


def _build_answer_prompt(query: str, results: List[Any], settings: Any):
    """Create a grounded QA prompt from retrieved chunks."""
    context_blocks = []
    for idx, result in enumerate(results, start=1):
        metadata = _result_metadata(result)
        source = metadata.get("source_path", "unknown")
        chunk_id = _result_field(result, "chunk_id", "")
        text = (_result_field(result, "text", "") or "").strip()
        if not text:
            continue
        context_blocks.append(
            f"[{idx}] source={source} chunk_id={chunk_id}\n{text}"
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "（无可用上下文）"
    return build_answer_messages(query, context_text, settings=settings)


def _generate_final_answer(settings: Any, query: str, results: List[Any]) -> str:
    """Generate final answer from retrieval results via configured LLM."""
    llm = LLMFactory.create(settings)
    messages = _build_answer_prompt(query, results, settings)
    response = llm.chat(messages)
    return response.content.strip()


def _build_answer_settings(
    settings: Any,
    answer_temperature: float,
    answer_max_tokens: int,
) -> Any:
    llm_override = dc_replace(
        settings.llm,
        temperature=answer_temperature,
        max_tokens=answer_max_tokens,
    )
    return dc_replace(settings, llm=llm_override)


def _render_sources_for_answer(results: List[Any]) -> None:
    st.markdown("**Sources**")
    for idx, result in enumerate(results, start=1):
        metadata = _result_metadata(result)
        source = metadata.get("source_path", "unknown")
        page_num = metadata.get("page_num", "")
        score = _result_field(result, "score", 0.0)
        page_info = f" · p.{page_num}" if page_num != "" else ""
        st.caption(f"[{idx}] {source}{page_info} · score={score:.4f}")


def _snapshot_results(results: List[Any]) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for result in results:
        snapshots.append(
            {
                "chunk_id": _result_field(result, "chunk_id", ""),
                "score": _result_field(result, "score", 0.0),
                "text": _result_field(result, "text", "") or "",
                "metadata": _result_metadata(result),
            }
        )
    return snapshots


def _preview_retrieval_list(results: Any, limit: int = 8) -> List[Dict[str, Any]]:
    if not results:
        return []
    previews: List[Dict[str, Any]] = []
    for r in results[:limit]:
        if isinstance(r, dict):
            meta = r.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            previews.append(
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "score": round(float(r.get("score", 0.0)), 4),
                    "text_preview": (r.get("text") or "")[:450],
                    "source": meta.get("source_path", meta.get("source", "")),
                }
            )
            continue
        meta = getattr(r, "metadata", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        previews.append(
            {
                "chunk_id": getattr(r, "chunk_id", "") or "",
                "score": round(float(getattr(r, "score", 0.0)), 4),
                "text_preview": (getattr(r, "text", "") or "")[:450],
                "source": meta.get("source_path", meta.get("source", "")),
            }
        )
    return previews


def _render_hit_preview_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.caption("无预览。")
        return
    for i, row in enumerate(rows, start=1):
        cid = row.get("chunk_id") or f"#{i}"
        with st.expander(f"{cid} · score={row.get('score', 0):.4f}", expanded=False):
            if row.get("source"):
                st.caption(str(row["source"]))
            st.text((row.get("text_preview") or "") + ("" if len(row.get("text_preview") or "") < 450 else "…"))


def _render_pipeline_section(pipeline: Any) -> None:
    if not isinstance(pipeline, dict) or not pipeline:
        return
    with st.expander(
        "检索管线（Query Processing → Dense / Sparse → Fusion → Rerank）",
        expanded=False,
    ):
        pq = pipeline.get("processed_query") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original query**")
            st.info(pq.get("original_query", "—"))
        with c2:
            st.markdown("**Keywords**")
            kws = pq.get("keywords") or []
            if kws:
                st.markdown(" · ".join(f"`{x}`" for x in kws))
            else:
                st.caption("—")
        flt = pq.get("filters") or {}
        if flt:
            st.markdown("**Parsed filters**")
            st.json(flt)
        st.caption(
            f"Dense hits: **{pipeline.get('dense_hits', 0)}** · "
            f"Sparse hits: **{pipeline.get('sparse_hits', 0)}** · "
            f"Fused: **{pipeline.get('fused_hits', 0)}** · "
            f"Rerank: **{'on' if pipeline.get('rerank_applied') else 'off'}**"
        )
        if pipeline.get("dense_error"):
            st.error(f"Dense: {pipeline['dense_error']}")
        if pipeline.get("sparse_error"):
            st.error(f"Sparse: {pipeline['sparse_error']}")
        if pipeline.get("used_fallback"):
            st.info("本次检索在 Dense/Sparse 之一失败时使用了单侧回退。")

        t_dense, t_sparse, t_fusion, t_rr = st.tabs(["Dense", "Sparse", "Fusion", "Rerank"])
        with t_dense:
            _render_hit_preview_rows(pipeline.get("dense_preview") or [])
        with t_sparse:
            _render_hit_preview_rows(pipeline.get("sparse_preview") or [])
        with t_fusion:
            _render_hit_preview_rows(pipeline.get("fused_preview") or [])
        with t_rr:
            if pipeline.get("rerank_applied"):
                st.caption(
                    f"输入 {pipeline.get('rerank_input', '—')} → "
                    f"输出 {pipeline.get('rerank_output', '—')} "
                    f"(rerank_top_k={pipeline.get('rerank_top_k_requested', '—')})"
                )
                _render_hit_preview_rows(pipeline.get("rerank_preview") or [])
            else:
                st.info("未应用 Rerank（未勾选或 reranker 未启用）。")


def _render_round_meta(round_data: Dict[str, Any], title: str) -> None:
    st.markdown(f"**{title}**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption(f"query: {round_data.get('query', '')}")
        st.caption(f"collection: {round_data.get('collection', 'default')}")
    with col2:
        st.caption(f"top_k(out): {round_data.get('top_k', 5)}")
        st.caption(f"top_k(retrieve): {round_data.get('retrieve_top_k', 5)}")
    with col3:
        st.caption(f"rerank: {'on' if round_data.get('use_rerank', False) else 'off'}")
        st.caption(f"rerank_top_k: {round_data.get('rerank_top_k', '-')}")
    with col4:
        st.caption(f"hits: {round_data.get('result_count', 0)}")
        st.caption(f"latency: {round_data.get('elapsed_ms', 0)} ms")
    st.caption(
        f"dense_top_k={round_data.get('dense_top_k', '-')} · "
        f"sparse_top_k={round_data.get('sparse_top_k', '-')} · "
        f"rrf_k={round_data.get('rrf_k', '-')} · "
        f"max_keywords={round_data.get('query_max_keywords', '-')} · "
        f"filter_parse={'on' if round_data.get('enable_filter_parsing', False) else 'off'}"
    )
    st.caption(f"time: {round_data.get('created_at', '-')}")


def _render_current_round(round_data: Dict[str, Any]) -> None:
    """Show the latest query round: answer first, optional chunks behind a toggle."""
    st.divider()
    st.subheader("本轮结果")
    _render_round_meta(round_data, "Query")
    _render_pipeline_section(round_data.get("pipeline"))

    answer = (round_data.get("answer") or "").strip()
    if answer:
        st.markdown("**Final Answer**")
        st.markdown(answer)
        _render_sources_for_answer(round_data.get("results", []))
    elif round_data.get("generate_answer"):
        st.info("未生成最终答案（可能未勾选或 LLM 调用失败），可展开检索片段查看上下文。")
    elif not round_data.get("results"):
        st.info("没有检索到结果。请检查 collection 或先导入文档。")

    btn1, btn2, _ = st.columns([1, 1, 4])
    with btn1:
        show = st.session_state.get("ask_show_chunks", False)
        label = "隐藏检索片段" if show else "显示检索片段"
        if st.button(label, key="ask_toggle_chunks"):
            st.session_state["ask_show_chunks"] = not show
            st.rerun()
    with btn2:
        if st.button("清除本轮", key="ask_clear_current"):
            st.session_state["ask_last_round"] = None
            st.session_state["ask_show_chunks"] = False
            st.rerun()

    if st.session_state.get("ask_show_chunks") and round_data.get("results"):
        _render_results(round_data["results"], key_prefix="current")


def _render_history() -> None:
    with st.expander("历史记录", expanded=False):
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("清空历史", key="ask_clear_history"):
                st.session_state["ask_round_history"] = []
                st.rerun()
        with col2:
            st.caption("过往提问与答案；默认折叠，不影响新一轮提问。")

        history = st.session_state.get("ask_round_history", [])
        if not history:
            st.caption("暂无历史。")
            return

        for idx, item in enumerate(reversed(history), start=1):
            summary = (
                f"Round {len(history) - idx + 1} · "
                f"{item.get('query', '')[:50]} · "
                f"hits={item.get('result_count', 0)} · "
                f"{item.get('elapsed_ms', 0)} ms"
            )
            with st.expander(summary, expanded=False):
                _render_round_meta(item, "Round Meta")
                _render_pipeline_section(item.get("pipeline"))
                hist_answer = item.get("answer", "")
                if hist_answer:
                    st.markdown("**Final Answer**")
                    st.markdown(hist_answer)
                    _render_sources_for_answer(item.get("results", []))
                if item.get("results"):
                    _render_results(item.get("results", []), key_prefix=f"history_{idx}")


def render() -> None:
    """Render the Ask page."""
    _ensure_ask_state()
    st.header("💬 Ask")
    st.caption(
        "提交后展示 **Final Answer** 与引用；**检索管线** 可展开查看各阶段预览；"
        "完整片段列表仍用「显示检索片段」展开。"
    )

    svc = DataService()
    collections = svc.list_collections() or []
    cfg_name = (load_settings().vector_store.collection_name or "default").strip()
    if cfg_name and cfg_name not in collections:
        collections.insert(0, cfg_name)
    non_empty = [c for c in collections if c != "default"]
    pick = non_empty[0] if non_empty else (collections[0] if collections else "default")
    default_idx = collections.index(pick) if pick in collections else 0
    settings = load_settings()

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_area(
            "Question",
            value=st.session_state.get("ask_last_query", ""),
            placeholder="例如：七年级下册里，一元一次方程有哪些典型题型？",
            height=90,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            collection = st.selectbox(
                "Collection",
                options=collections if collections else ["default"],
                index=min(default_idx, max(len(collections) - 1, 0)),
            )
        with col2:
            top_k = st.slider(
                "Top K (output)",
                min_value=1,
                max_value=20,
                value=5,
                help="最终展示/返回的结果数。",
            )
        with col3:
            use_rerank = st.checkbox("Use rerank", value=True)
        generate_answer = st.checkbox("Generate final answer", value=True)
        with st.expander("Advanced Retrieval Parameters", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                retrieve_top_k = st.number_input(
                    "Retrieve Top-K (pre-rerank)",
                    min_value=1,
                    max_value=50,
                    value=max(int(top_k), int(settings.retrieval.fusion_top_k)),
                    step=1,
                )
                dense_top_k = st.number_input(
                    "Dense Top-K",
                    min_value=1,
                    max_value=100,
                    value=int(settings.retrieval.dense_top_k),
                    step=1,
                )
                sparse_top_k = st.number_input(
                    "Sparse Top-K",
                    min_value=1,
                    max_value=100,
                    value=int(settings.retrieval.sparse_top_k),
                    step=1,
                )
            with col_b:
                rrf_k = st.number_input(
                    "RRF k",
                    min_value=1,
                    max_value=300,
                    value=int(settings.retrieval.rrf_k),
                    step=1,
                )
                rerank_top_k = st.number_input(
                    "Rerank Top-K",
                    min_value=1,
                    max_value=50,
                    value=int(top_k),
                    step=1,
                    help="重排后保留的结果数。",
                )
            with col_c:
                query_max_keywords = st.number_input(
                    "Query Max Keywords",
                    min_value=1,
                    max_value=50,
                    value=20,
                    step=1,
                )
                enable_filter_parsing = st.checkbox(
                    "Enable filter parsing (collection:xx)",
                    value=True,
                )
                answer_temperature = st.slider(
                    "Answer Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(settings.llm.temperature),
                    step=0.1,
                    help="只影响最终答案生成，不影响检索。",
                )
                answer_max_tokens = st.number_input(
                    "Answer Max Tokens",
                    min_value=128,
                    max_value=8192,
                    value=int(settings.llm.max_tokens),
                    step=128,
                    help="只影响最终答案生成，不影响检索。",
                )
        submitted = st.form_submit_button("Ask")

    last_round = st.session_state.get("ask_last_round")
    if last_round and not submitted:
        _render_current_round(last_round)

    if not submitted:
        _render_history()
        return

    if not query.strip():
        st.warning("请输入问题后再查询。")
        return

    st.session_state["ask_last_query"] = query

    trace = TraceContext(trace_type="query")
    trace.metadata["source"] = "dashboard_ask"
    trace.metadata["query"] = query[:200]
    trace.metadata["collection"] = collection
    trace.metadata["top_k"] = top_k
    trace.metadata["retrieve_top_k"] = int(retrieve_top_k)
    trace.metadata["dense_top_k"] = int(dense_top_k)
    trace.metadata["sparse_top_k"] = int(sparse_top_k)
    trace.metadata["rrf_k"] = int(rrf_k)
    trace.metadata["use_rerank"] = use_rerank
    trace.metadata["rerank_top_k"] = int(rerank_top_k)
    effective_query = query
    trace.metadata["conversation_mode"] = False
    trace.metadata["effective_query_preview"] = query[:300]

    with st.spinner("Retrieving..."):
        try:
            hybrid_search, reranker = _build_components(
                settings=settings,
                collection=collection,
                query_max_keywords=int(query_max_keywords),
                enable_filter_parsing=enable_filter_parsing,
            )
            hybrid_search.config.dense_top_k = int(dense_top_k)
            hybrid_search.config.sparse_top_k = int(sparse_top_k)
            hybrid_search.config.fusion_top_k = int(retrieve_top_k)
            from src.core.query_engine.fusion import RRFFusion
            hybrid_search.fusion = RRFFusion(k=int(rrf_k))
            t0 = time.monotonic()
            hybrid_detail = hybrid_search.search(
                query=effective_query,
                top_k=int(retrieve_top_k),
                filters=None,
                trace=trace,
                return_details=True,
            )
            if not isinstance(hybrid_detail, HybridSearchResult):
                raise TypeError("internal: expected HybridSearchResult")
            fused_list = list(hybrid_detail.results)
            pq = hybrid_detail.processed_query
            pipeline: Dict[str, Any] = {
                "processed_query": pq.to_dict(),
                "dense_error": hybrid_detail.dense_error,
                "sparse_error": hybrid_detail.sparse_error,
                "used_fallback": hybrid_detail.used_fallback,
                "dense_hits": len(hybrid_detail.dense_results or []),
                "sparse_hits": len(hybrid_detail.sparse_results or []),
                "fused_hits": len(fused_list),
                "dense_preview": _preview_retrieval_list(hybrid_detail.dense_results),
                "sparse_preview": _preview_retrieval_list(hybrid_detail.sparse_results),
                "fused_preview": _preview_retrieval_list(fused_list),
            }
            if use_rerank and reranker.is_enabled and fused_list:
                rerank_result = reranker.rerank(
                    query=query,
                    results=fused_list,
                    top_k=int(rerank_top_k),
                    trace=trace,
                )
                results = list(rerank_result.results)[: int(top_k)]
                pipeline["rerank_applied"] = True
                pipeline["rerank_input"] = len(fused_list)
                pipeline["rerank_output"] = len(rerank_result.results)
                pipeline["rerank_top_k_requested"] = int(rerank_top_k)
                pipeline["rerank_preview"] = _preview_retrieval_list(rerank_result.results)
            else:
                results = fused_list[: int(top_k)]
                pipeline["rerank_applied"] = False
            elapsed_ms = (time.monotonic() - t0) * 1000.0
        except Exception as e:
            st.error(f"Query failed: {e}")
            TraceCollector().collect(trace)
            return

    TraceCollector().collect(trace)
    st.success(f"Done in {elapsed_ms:.0f} ms")
    round_data: Dict[str, Any] = {
        "query": query,
        "effective_query": effective_query,
        "collection": collection,
        "top_k": top_k,
        "retrieve_top_k": int(retrieve_top_k),
        "use_rerank": use_rerank,
        "rerank_top_k": int(rerank_top_k),
        "dense_top_k": int(dense_top_k),
        "sparse_top_k": int(sparse_top_k),
        "rrf_k": int(rrf_k),
        "query_max_keywords": int(query_max_keywords),
        "enable_filter_parsing": enable_filter_parsing,
        "answer_temperature": float(answer_temperature),
        "answer_max_tokens": int(answer_max_tokens),
        "generate_answer": generate_answer,
        "result_count": len(results),
        "elapsed_ms": round(elapsed_ms),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": _snapshot_results(results),
        "answer": "",
        "pipeline": pipeline,
    }
    answer_text = ""
    if results and generate_answer:
        with st.spinner("Generating final answer..."):
            try:
                answer_settings = _build_answer_settings(
                    settings=settings,
                    answer_temperature=float(answer_temperature),
                    answer_max_tokens=int(answer_max_tokens),
                )
                answer_text = _generate_final_answer(answer_settings, query, results)
            except Exception as e:
                st.warning(f"生成最终答案失败，将仅展示检索片段：{e}")
                answer_text = ""

    round_data["answer"] = answer_text
    st.session_state["ask_round_history"].append(round_data)
    st.session_state["ask_last_round"] = round_data
    st.session_state["ask_show_chunks"] = not bool(answer_text)

    _render_current_round(round_data)
    _render_history()
