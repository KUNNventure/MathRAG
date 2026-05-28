"""Evaluation Panel page – run evaluations and view metrics.

Layout (tabs):
1. Reports: load report.json, aggregate + per-query metrics, export JSON
2. Run: golden set, retrieval overrides, optional per-case answer overrides, Ragas judge model
3. History: JSONL runs table, metric trends, reload a run into the report viewer
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Default golden test set location
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
# Evaluation results history file (relative → resolve_path in _eval_history_path)
EVAL_HISTORY_REL = Path("logs/eval_history.jsonl")


def _eval_history_path() -> Path:
    from src.core.settings import resolve_path
    return resolve_path(EVAL_HISTORY_REL)


def _resolve_path_str(path_str: str) -> Path:
    raw = (path_str or "").strip()
    if not raw or raw in (".", ".."):
        return DEFAULT_GOLDEN_SET.resolve()
    p = Path(raw)
    if not p.is_absolute():
        from src.core.settings import resolve_path
        return resolve_path(p)
    return p


def _discover_report_files() -> List[Path]:
    from src.observability.dashboard.services.eval_results_service import (
        list_experiment_dirs,
    )

    found: List[Path] = []
    for d in list_experiment_dirs():
        rp = d / "report.json"
        if rp.is_file():
            found.append(rp)
    return sorted(found, key=lambda p: p.parent.name, reverse=True)


def render() -> None:
    """Render the Evaluation Panel page."""
    st.header("📏 Evaluation Panel")
    st.caption(
        "完整评测台：**报告与指标**（加载 JSON）、**运行评测**（Golden + Hybrid + Ragas）、"
        "**历史与对比**（JSONL 记录与指标趋势）。运行评测会产生 API 费用。"
    )

    tab_reports, tab_experiments, tab_run, tab_hist = st.tabs(
        ["📂 报告与指标", "🗂️ 全部实验", "▶️ 运行评测", "📈 历史与对比"],
    )
    with tab_reports:
        _render_reports_tab()
    with tab_experiments:
        _render_experiments_tab()
    with tab_run:
        _render_run_evaluation_form()
    with tab_hist:
        _render_history_tab()


def _render_experiments_tab() -> None:
    """Browse every results/<run_id>/ directory with metrics, tokens, and file links."""
    from src.observability.dashboard.services.eval_results_service import (
        list_experiment_dirs,
        load_master_summary,
        save_master_summary,
        summarize_experiment,
    )

    st.subheader("全部评测实验（results/）")
    st.caption(
        "扫描各实验目录的 report.json / config.json / answers.jsonl，"
        "含 RAGAS、耗时、切片与 token 粗估。"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("刷新实验索引", type="primary", key="eval_refresh_master"):
            save_master_summary()
            st.success("已写入 results/_MASTER_SUMMARY.json")
    with c2:
        master = load_master_summary()
        if master:
            st.caption(f"索引中 **{master.get('experiment_count', 0)}** 个实验")

    dirs = list_experiment_dirs()
    if not dirs:
        st.warning("未找到 results/*/report.json，请先运行 scripts/evaluate.py。")
        return

    summaries = [summarize_experiment(d) for d in dirs]
    table_rows: List[Dict[str, Any]] = []
    for s in summaries:
        tok = s.token_estimate or {}
        table_rows.append(
            {
                "实验": s.run_id,
                "题数": s.query_count,
                "collection": s.collection,
                "chunk": s.chunk_size,
                "top_k": s.top_k,
                "CR": s.context_recall,
                "CP": s.context_precision,
                "Faith": s.faithfulness,
                "Score_p": s.score_partial,
                "Score": s.score_full,
                "耗时(s)": s.elapsed_seconds,
                "token估": tok.get("total_tokens_est"),
                "费用估¥": tok.get("cost_est_cny"),
                "faith_only": s.faith_only,
                "cr_cp_only": s.cr_cp_only,
            }
        )
    st.dataframe(table_rows, use_container_width=True, height=min(520, 80 + len(table_rows) * 36))

    run_ids = [s.run_id for s in summaries]
    pick_id = st.selectbox("查看单次实验详情", options=run_ids, key="eval_exp_pick")
    picked = next((s for s in summaries if s.run_id == pick_id), None)
    if not picked:
        return

    st.markdown(f"### `{pick_id}`")
    mcols = st.columns(6)
    metrics = [
        ("CR", picked.context_recall),
        ("CP", picked.context_precision),
        ("Faith", picked.faithfulness),
        ("Score_p", picked.score_partial),
        ("Score", picked.score_full),
        ("耗时(s)", picked.elapsed_seconds),
    ]
    for i, (label, val) in enumerate(metrics):
        with mcols[i % 6]:
            st.metric(label, f"{val:.4f}" if isinstance(val, (int, float)) else "—")

    if picked.token_estimate:
        st.markdown("**Token 粗估**")
        st.json(picked.token_estimate)

    fcols = st.columns(3)
    with fcols[0]:
        if picked.report_path:
            st.code(picked.report_path, language=None)
    with fcols[1]:
        if picked.config_path:
            st.code(picked.config_path, language=None)
    with fcols[2]:
        if picked.answers_path:
            st.code(picked.answers_path, language=None)

    if st.button("加载该实验 report 到「报告与指标」", key="eval_exp_load_report"):
        rp = Path(picked.report_path) if picked.report_path else None
        if rp and rp.is_file():
            report_view = json.loads(rp.read_text(encoding="utf-8"))
            st.session_state["eval_loaded_report"] = report_view
            st.session_state["eval_loaded_report_path"] = str(rp)
            st.success("已加载，请切换到 **报告与指标** 查看逐题明细。")
        else:
            st.error("无 report.json")

    with st.expander("answers.jsonl 预览（前 3 条）"):
        ap = Path(picked.answers_path) if picked.answers_path else None
        if ap and ap.is_file():
            lines = ap.read_text(encoding="utf-8").strip().splitlines()[:3]
            for line in lines:
                st.json(json.loads(line))
        else:
            st.info("无 answers.jsonl")


def _render_reports_tab() -> None:
    """Load and display a saved report.json (read-only)."""
    st.subheader("查看评测报告（只读）")
    reports = _discover_report_files()
    report_options = [str(p) for p in reports] if reports else []
    default_report = "results/baseline/report.json"
    if default_report not in report_options and Path(default_report).is_file():
        report_options.insert(0, default_report)
    if not report_options:
        report_options = [default_report]

    pick = st.selectbox(
        "选择 report.json",
        options=report_options,
        index=0,
        key="eval_report_pick",
    )
    manual_path = st.text_input(
        "或手动输入路径",
        value=pick,
        key="eval_view_report_path",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load report", type="primary", key="eval_load_report_btn"):
            rp = _resolve_path_str(manual_path)
            if not rp.is_file():
                st.error(f"文件不存在: `{rp}`")
            else:
                try:
                    report_view = json.loads(rp.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    st.error(f"JSON 解析失败: {exc}")
                else:
                    st.session_state["eval_loaded_report"] = report_view
                    st.session_state["eval_loaded_report_path"] = str(rp)
                    st.success(f"已加载: `{rp}`")
    with c2:
        loaded = st.session_state.get("eval_loaded_report")
        if isinstance(loaded, dict) and loaded:
            try:
                blob = json.dumps(loaded, ensure_ascii=False, indent=2)
                st.download_button(
                    label="下载当前报告 JSON",
                    data=blob,
                    file_name="eval_report_export.json",
                    mime="application/json",
                    key="eval_dl_report",
                )
            except Exception:
                pass

    loaded = st.session_state.get("eval_loaded_report")
    if loaded:
        st.caption(f"当前报告: `{st.session_state.get('eval_loaded_report_path', '')}`")
        _render_aggregate_metrics(loaded)
        _render_query_details(loaded)


def _render_run_evaluation_form() -> None:
    """Configuration + run button for a new evaluation job."""
    st.subheader("⚙️ Configuration")

    col1, col2, col3 = st.columns(3)
    try:
        from src.core.settings import load_settings
        settings = load_settings()
    except Exception:
        settings = None

    with col1:
        backend = st.selectbox(
            "Evaluator Backend",
            options=["ragas", "custom", "composite"],
            index=0,
            key="eval_backend",
            help="Select which evaluator backend to use.",
        )

    # Show info/warning based on selected backend
    if backend in ("custom", "composite"):
        st.info(
            "ℹ️ **Custom Evaluator** 尚未完成数据集准备，当前仅为预留接口。"
            "Custom Evaluator 需要在 Golden Test Set 中填写 `expected_chunk_ids` "
            "作为 ground truth 才能计算 hit_rate / MRR 指标。"
            "目前建议使用 **ragas** 后端进行评估。",
            icon="🚧",
        )

    with col2:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=50,
            value=10,
            key="eval_top_k",
            help="Number of chunks to retrieve per query.",
        )

    with col3:
        try:
            from src.observability.dashboard.services.data_service import DataService as _DS
            _coll_choices = _DS().list_collections() or []
        except Exception:
            _coll_choices = []
        _eval_coll_opts = [""] + sorted(set(_coll_choices))
        collection = st.selectbox(
            "Collection",
            options=_eval_coll_opts,
            index=0,
            key="eval_collection_select",
            format_func=lambda x: "— (settings / empty) —" if x == "" else x,
            help="与 Data Browser / Ask 使用相同 collection 名称；空则交由 EvalRunner 默认。",
        )

    golden_path_str = st.text_input(
        "Golden Test Set Path",
        value=str(DEFAULT_GOLDEN_SET.resolve()),
        key="eval_golden_path",
        help="Path to the golden_test_set.json file.",
    )
    golden_path = _resolve_path_str(golden_path_str)

    with st.expander("Advanced Retrieval Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            dense_top_k = st.number_input(
                "Dense Top-K",
                min_value=1,
                max_value=100,
                value=int(settings.retrieval.dense_top_k) if settings else 20,
                step=1,
            )
            sparse_top_k = st.number_input(
                "Sparse Top-K",
                min_value=1,
                max_value=100,
                value=int(settings.retrieval.sparse_top_k) if settings else 20,
                step=1,
            )
        with c2:
            rrf_k = st.number_input(
                "RRF k",
                min_value=1,
                max_value=300,
                value=int(settings.retrieval.rrf_k) if settings else 60,
                step=1,
            )
            enable_rerank = st.checkbox(
                "Enable rerank",
                value=bool(settings.rerank.enabled) if settings else True,
            )
        with c3:
            rerank_top_k = st.number_input(
                "Rerank Top-K",
                min_value=1,
                max_value=50,
                value=int(settings.rerank.top_k) if settings else 5,
                step=1,
            )

    if not golden_path.is_file():
        st.warning(
            f"⚠️ **Golden test set not found:** `{golden_path}`. "
            "See `tests/fixtures/golden_test_set.json` for the format."
        )
    else:
        try:
            n_cases = len(_load_golden_queries(golden_path))
            st.caption(f"Golden set: **{n_cases}** test cases · `{golden_path}`")
        except Exception as exc:
            st.warning(f"无法读取 golden set: {exc}")

    st.caption(
        "默认由 **EvalRunner** 按检索结果自动生成回答；可在下方可选覆盖逐题答案。"
        "Golden 中的 `reference_answer` 可作为对照填入「逐题覆盖」。"
    )

    cases_for_ui: List[Dict[str, Any]] = []
    if golden_path.is_file():
        try:
            cases_for_ui = _load_golden_queries(golden_path)
        except Exception:
            cases_for_ui = []

    with st.expander("逐题覆盖 Generated Answer（可选）", expanded=False):
        if not cases_for_ui:
            st.caption("无法读取 golden 或未配置用例。")
        else:
            st.caption("仅非空项会覆盖自动生成的回答（最多展示前 40 条）。")
            for i, tc in enumerate(cases_for_ui[:40]):
                qprev = (tc.get("query") or "")[:80]
                st.text_area(
                    f"Case {i + 1}",
                    value=tc.get("reference_answer") or "",
                    height=64,
                    key=f"eval_ov_ans_{i}",
                    help=qprev,
                )

    st.text_input(
        "Ragas Judge 模型覆盖（可选）",
        value="",
        key="eval_judge_model",
        help="仅对 ragas 后端传入 EvaluatorFactory；留空使用 settings 默认。",
    )

    confirm_cost = st.checkbox(
        "我已知晓本次运行可能产生 API 费用",
        value=False,
        key="eval_cost_confirm",
    )

    run_clicked = st.button(
        "▶️  Run Evaluation",
        type="primary",
        key="eval_run_btn",
        disabled=not golden_path.is_file() or not confirm_cost,
    )

    if run_clicked:
        overrides: Dict[int, str] = {}
        for i in range(min(len(cases_for_ui), 40)):
            k = f"eval_ov_ans_{i}"
            if k in st.session_state:
                val = str(st.session_state[k]).strip()
                if val:
                    overrides[i] = val
        judge_raw = (st.session_state.get("eval_judge_model") or "").strip()
        judge_model = judge_raw or None
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
            user_answers=overrides or None,
            dense_top_k=int(dense_top_k),
            sparse_top_k=int(sparse_top_k),
            rrf_k=int(rrf_k),
            enable_rerank=enable_rerank,
            rerank_top_k=int(rerank_top_k),
            judge_model=judge_model if backend == "ragas" else None,
        )


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    rrf_k: int = 60,
    enable_rerank: bool = True,
    rerank_top_k: int = 5,
    judge_model: Optional[str] = None,
) -> None:
    """Execute an evaluation run and display results.

    Attempts to load the evaluator, run the golden test set, and
    display aggregate + per-query metrics.  Falls back to a graceful
    error message on failure.
    """
    with st.spinner("Loading evaluator and running evaluation…"):
        try:
            report_dict = _execute_evaluation(
                backend=backend,
                golden_path=golden_path,
                top_k=top_k,
                collection=collection,
                user_answers=user_answers,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                rrf_k=rrf_k,
                enable_rerank=enable_rerank,
                rerank_top_k=rerank_top_k,
                judge_model=judge_model,
            )
        except Exception as exc:
            st.error(f"❌ Evaluation failed: {exc}")
            logger.exception("Evaluation failed")
            return

    # ── Display results ────────────────────────────────────────────
    st.success("✅ Evaluation complete!")

    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)

    st.session_state["eval_loaded_report"] = report_dict
    st.session_state["eval_loaded_report_path"] = "«最近一次运行»"

    # Save to history
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    rrf_k: int = 60,
    enable_rerank: bool = True,
    rerank_top_k: int = 5,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline and return the report dict.

    This function imports heavy dependencies lazily to keep the
    dashboard responsive when the page is not used.
    """
    from dataclasses import replace as dc_replace

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner, load_test_set

    settings = load_settings()
    overridden_retrieval = type(settings.retrieval)(
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        fusion_top_k=top_k,
        rrf_k=rrf_k,
    )
    overridden_rerank = type(settings.rerank)(
        enabled=enable_rerank,
        provider=settings.rerank.provider,
        model=settings.rerank.model,
        top_k=rerank_top_k,
    )
    settings = dc_replace(
        settings,
        retrieval=overridden_retrieval,
        rerank=overridden_rerank,
    )

    # Override evaluator provider from UI selection — build a new full
    # Settings object so that RagasEvaluator can still access .llm / .embedding.
    eval_settings = settings.evaluation
    overridden_eval = type(eval_settings)(
        enabled=True,
        provider=backend,
        metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
    )
    # Replace only the evaluation sub-config in the full settings
    settings_with_override = dc_replace(settings, evaluation=overridden_eval)

    factory_kwargs: Dict[str, Any] = {}
    if judge_model:
        factory_kwargs["judge_model"] = judge_model
    evaluator = EvaluatorFactory.create(settings_with_override, **factory_kwargs)

    # Try to create HybridSearch (optional – works without if not configured)
    target_collection = collection or "default"
    hybrid_search = _try_create_hybrid_search(settings, target_collection)

    # Create reranker if enabled
    reranker = None
    try:
        from src.core.query_engine.reranker import create_core_reranker
        reranker = create_core_reranker(settings=settings)
        if not reranker.is_enabled:
            reranker = None
    except Exception as exc:
        logger.warning("Could not create reranker: %s", exc)

    # Build answer_override map: index → user-provided answer text
    # EvalRunner will use these instead of auto-generating from chunks.
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_overrides=user_answers,
        reranker=reranker,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=collection,
    )

    report_dict = report.to_dict()
    report_dict["runtime_params"] = {
        "dense_top_k": dense_top_k,
        "sparse_top_k": sparse_top_k,
        "rrf_k": rrf_k,
        "enable_rerank": enable_rerank,
        "rerank_top_k": rerank_top_k,
        "top_k": top_k,
        "collection": collection or "default",
    }
    return report_dict


def _try_create_hybrid_search(settings: Any, collection: str = "default") -> Any:
    """Attempt to create a HybridSearch instance.

    Returns None if required dependencies are not available
    (e.g., no indexed data).
    """
    try:
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        return create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None


def _render_aggregate_metrics(report: Dict[str, Any]) -> None:
    """Display aggregate metrics as metric cards."""
    st.subheader("📊 Aggregate Metrics")

    agg = report.get("aggregate_metrics", {})

    if not agg:
        st.info("No aggregate metrics available.")
        return

    cols = st.columns(min(len(agg), 4))
    for idx, (name, value) in enumerate(sorted(agg.items())):
        with cols[idx % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )

    st.caption(
        f"Evaluator: **{report.get('evaluator_name', '—')}** · "
        f"Queries: **{report.get('query_count', 0)}** · "
        f"Total time: **{report.get('total_elapsed_ms', 0):.0f} ms**"
    )
    runtime_params = report.get("runtime_params", {})
    if runtime_params:
        st.caption(
            "Params: "
            f"dense_top_k={runtime_params.get('dense_top_k', '-')} · "
            f"sparse_top_k={runtime_params.get('sparse_top_k', '-')} · "
            f"rrf_k={runtime_params.get('rrf_k', '-')} · "
            f"rerank={'on' if runtime_params.get('enable_rerank', False) else 'off'} · "
            f"rerank_top_k={runtime_params.get('rerank_top_k', '-')} · "
            f"top_k={runtime_params.get('top_k', '-')} · "
            f"collection={runtime_params.get('collection', '-')}"
        )


def _render_query_details(report: Dict[str, Any]) -> None:
    """Display per-query evaluation results in an expandable table."""
    st.subheader("🔍 Per-Query Details")

    query_results = report.get("query_results", [])
    if not query_results:
        st.info("No per-query results available.")
        return

    for idx, qr in enumerate(query_results):
        query = qr.get("query", "—")
        elapsed = qr.get("elapsed_ms", 0)
        metrics = qr.get("metrics", {})

        # Build metric summary for the expander label
        metric_summary = " · ".join(
            f"{k}: {v:.3f}" for k, v in sorted(metrics.items())
        )
        if not metric_summary:
            metric_summary = "no metrics"

        with st.expander(
            f"**Q{idx + 1}**: {query[:80]} — {elapsed:.0f} ms — {metric_summary}",
            expanded=False,
        ):
            # Metrics
            if metrics:
                mcols = st.columns(min(len(metrics), 4))
                for midx, (mname, mval) in enumerate(sorted(metrics.items())):
                    with mcols[midx % len(mcols)]:
                        st.metric(mname, f"{mval:.4f}")

            # Retrieved chunks
            chunks = qr.get("retrieved_chunk_ids", [])
            if chunks:
                st.markdown(f"**Retrieved Chunks** ({len(chunks)}):")
                st.code(", ".join(chunks[:20]), language=None)

            # Generated answer
            answer = qr.get("generated_answer")
            if answer:
                st.markdown("**Generated Answer:**")
                st.text(answer[:500])


def _render_history_tab() -> None:
    """Historical evaluation runs, trend chart, and reload into report tab."""
    st.subheader("评测运行历史")

    history = _load_history()
    if not history:
        st.info(
            "暂无历史记录。在 **运行评测** 标签完成一次评测后，会写入 "
            f"`{_eval_history_path()}`。"
        )
        return

    rows = []
    for entry in history[-25:]:
        runtime_params = entry.get("runtime_params", {})
        rows.append(
            {
                "Timestamp": entry.get("timestamp", "—"),
                "Evaluator": entry.get("evaluator_name", "—"),
                "Queries": entry.get("query_count", 0),
                "Time (ms)": round(entry.get("total_elapsed_ms", 0)),
                "Dense K": runtime_params.get("dense_top_k", "—"),
                "Sparse K": runtime_params.get("sparse_top_k", "—"),
                "RRF K": runtime_params.get("rrf_k", "—"),
                "Rerank": runtime_params.get("enable_rerank", "—"),
                "Rerank K": runtime_params.get("rerank_top_k", "—"),
                "Top K": runtime_params.get("top_k", "—"),
                **{
                    k: round(v, 4)
                    for k, v in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    st.dataframe(rows, use_container_width=True, height=min(460, 140 + len(rows) * 36))

    tail = history[-20:]
    metric_keys: set = set()
    for e in tail:
        metric_keys.update((e.get("aggregate_metrics") or {}).keys())
    if len(tail) > 1 and metric_keys:
        st.markdown("**聚合指标趋势（最近 {} 次）**".format(len(tail)))
        chart: Dict[str, List[float]] = {k: [] for k in sorted(metric_keys)}
        for e in tail:
            am = e.get("aggregate_metrics") or {}
            for k in sorted(metric_keys):
                chart[k].append(float(am.get(k) or 0.0))
        st.line_chart(chart)

    st.divider()
    st.markdown("**回放**")
    rev = list(reversed(history))
    pick = st.selectbox(
        "选择一条历史记录",
        options=list(range(len(rev))),
        format_func=lambda i: (
            f"{rev[i].get('timestamp', '—')} · {rev[i].get('evaluator_name', '—')} · "
            f"{rev[i].get('query_count', 0)} queries"
        ),
        key="eval_hist_pick",
    )
    if st.button("加载到「报告与指标」视图", key="eval_hist_load_btn"):
        entry = rev[pick]
        report_only = {k: v for k, v in entry.items() if k != "timestamp"}
        st.session_state["eval_loaded_report"] = report_only
        st.session_state["eval_loaded_report_path"] = (
            f"history:{entry.get('timestamp', '')}"
        )
        st.success("已写入会话，请切换到 **报告与指标** 查看。")


def _save_to_history(report: Dict[str, Any]) -> None:
    """Append an evaluation report to the history file."""
    try:
        hp = _eval_history_path()
        hp.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with hp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    """Load evaluation history from JSONL file."""
    hp = _eval_history_path()
    if not hp.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with hp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries


def _load_golden_queries(golden_path: Path) -> List[Dict[str, Any]]:
    """Load test cases from golden test set for display in the UI.

    Returns list of dicts with at least 'query' and optionally
    'reference_answer' keys.
    """
    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])
