"""Ingestion Manager page – upload files, local path batch, trigger ingestion, delete documents.

Layout:
1. File uploader + collection selector
2. Local directory / file path batch (scan + ingest)
3. Ingest button → progress bar (using on_progress callback)
4. Document list with delete buttons
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.services.data_service import DataService

HIERARCHICAL_FIXED_CHUNK_SIZE = 1000
HIERARCHICAL_FIXED_CHUNK_OVERLAP = 200

# Collection name suffix → (chunk_size, chunk_overlap). Longer suffixes first so `_c1500` wins over `_c500`.
_COLLECTION_CHUNK_BY_SUFFIX: tuple[tuple[str, int, int], ...] = (
    ("_c1500", 1500, 300),
    ("_c1000", 1000, 200),
    ("_c500", 500, 100),
)

# Sentinel option: pick this to show a text field for a brand-new collection name.
_NEW_COLLECTION_OPTION = "➕ New collection (type name below)"


def _locked_chunk_params_for_collection(collection_name: str) -> tuple[int, int] | None:
    """If collection name matches experiment suffixes, return fixed (chunk_size, overlap).

    Dashboard math-textbook runs use names like ``math_textbooks_c500``; keep overlap
    consistent with the tier without relying on manual Advanced field edits.
    """
    n = (collection_name or "").strip().lower()
    if not n:
        return None
    for suffix, size, overlap in _COLLECTION_CHUNK_BY_SUFFIX:
        if n.endswith(suffix):
            return (size, overlap)
    if n in ("c500", "c1000", "c1500"):
        return {"c500": (500, 100), "c1000": (1000, 200), "c1500": (1500, 300)}[n]
    return None


def _short_path(path_str: str, *, max_name: int = 72) -> str:
    if not path_str:
        return "—"
    name = Path(path_str).name
    if len(name) <= max_name:
        return name
    return name[: max_name - 1] + "…"


def _chroma_collection_names(data_svc: DataService | None) -> List[str]:
    """Return Chroma collection names for the ingest target picker."""
    if data_svc is None:
        return []
    try:
        raw = data_svc.list_collections()
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        return []


def _ingest_collection_select_options(
    chroma_names: List[str],
    configured_default: str,
) -> List[str]:
    """Ordered options: configured default first, then others, then 'new' sentinel."""
    seen: set[str] = set()
    ordered: List[str] = []
    cfg = (configured_default or "default").strip() or "default"

    for n in [cfg] + sorted(chroma_names):
        if n == _NEW_COLLECTION_OPTION or not n or n in seen:
            continue
        seen.add(n)
        ordered.append(n)

    if "default" not in seen:
        ordered.insert(0, "default")

    ordered.append(_NEW_COLLECTION_OPTION)
    return ordered


def _build_runtime_settings(base_settings: Any, params: Dict[str, Any]) -> Any:
    """Create a runtime settings object from UI overrides."""
    ingestion = base_settings.ingestion
    chunk_refiner = dict(getattr(ingestion, "chunk_refiner", {}) or {})
    metadata_enricher = dict(getattr(ingestion, "metadata_enricher", {}) or {})

    transform_llm_enabled = bool(params.get("transform_llm_enabled", True))
    refiner_llm = bool(params.get("chunk_refiner_use_llm", True)) and transform_llm_enabled
    enricher_llm = bool(params.get("metadata_enricher_use_llm", True)) and transform_llm_enabled

    chunk_refiner["use_llm"] = refiner_llm
    metadata_enricher["use_llm"] = enricher_llm

    overridden_ingestion = dc_replace(
        ingestion,
        chunk_size=int(params["chunk_size"]),
        chunk_overlap=int(params["chunk_overlap"]),
        splitter=str(params["splitter"]),
        batch_size=int(params["batch_size"]),
        chunk_refiner=chunk_refiner,
        metadata_enricher=metadata_enricher,
    )

    settings = dc_replace(base_settings, ingestion=overridden_ingestion)

    if hasattr(base_settings, "vision_llm") and base_settings.vision_llm is not None:
        settings = dc_replace(
            settings,
            vision_llm=dc_replace(
                base_settings.vision_llm,
                enabled=bool(params.get("image_caption_enabled", True)),
            ),
        )

    return settings


def _ingest_file_core(
    file_path: str,
    display_name: str,
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    params: Dict[str, Any],
    file_index: int,
    total_files: int,
) -> bool:
    """Run ingestion pipeline for one file on disk (temp or permanent path).

    Returns:
        True if ingestion succeeded, False otherwise.
    """
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    base_settings = load_settings()
    settings = _build_runtime_settings(base_settings, params)

    transform_mode = "LLM refine + enrich" if params.get("transform_llm_enabled", True) else "rule-based refine + enrich"
    if params.get("image_caption_enabled", False):
        transform_mode = f"{transform_mode} + image caption"

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": f"🔄 Transforming chunks ({transform_mode})…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        stage_frac = (current - 1) / total
        file_base = (file_index - 1) / total_files
        frac = file_base + (stage_frac / total_files)
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(
            frac,
            text=(
                f"[{file_index}/{total_files}] {display_name} · "
                f"[{current}/{total}] {label}"
            ),
        )
        status_text.caption(f"Processing {display_name} · {label}")

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = display_name
    trace.metadata["resolved_file_path"] = str(Path(file_path).resolve())
    trace.metadata["collection"] = collection
    doc_meta = params.get("ingest_document_metadata") or {}
    if isinstance(doc_meta, dict):
        trace.metadata["grade"] = doc_meta.get("grade")
        trace.metadata["volume"] = doc_meta.get("volume")
    trace.metadata["source"] = "dashboard"
    trace.metadata["batch_file_index"] = file_index
    trace.metadata["batch_total_files"] = total_files
    trace.metadata["runtime_ingestion_params"] = {
        "chunk_size": settings.ingestion.chunk_size if settings.ingestion else None,
        "chunk_overlap": settings.ingestion.chunk_overlap if settings.ingestion else None,
        "splitter": settings.ingestion.splitter if settings.ingestion else None,
        "batch_size": settings.ingestion.batch_size if settings.ingestion else None,
        "chunk_refiner_use_llm": (
            (settings.ingestion.chunk_refiner or {}).get("use_llm", False)
            if settings.ingestion else False
        ),
        "metadata_enricher_use_llm": (
            (settings.ingestion.metadata_enricher or {}).get("use_llm", False)
            if settings.ingestion else False
        ),
        "image_caption_enabled": (
            bool(settings.vision_llm.enabled)
            if getattr(settings, "vision_llm", None) is not None else False
        ),
        "force_reingest": bool(params.get("force_reingest", False)),
        "ingest_document_metadata": params.get("ingest_document_metadata"),
    }

    try:
        pipeline = IngestionPipeline(
            settings,
            collection=collection,
            force=bool(params.get("force_reingest", False)),
            document_metadata=params.get("ingest_document_metadata"),
        )
        result = pipeline.run(
            file_path=file_path,
            trace=trace,
            on_progress=on_progress,
        )
        integrity_stage = result.stages.get("integrity", {}) if hasattr(result, "stages") else {}
        if integrity_stage.get("skipped"):
            status_text.info(
                f"{display_name} skipped: same file hash already ingested "
                f"(collection={collection}). Enable 'Force Re-ingest' to reprocess."
            )
        file_done_frac = file_index / total_files
        progress_bar.progress(
            file_done_frac,
            text=f"[{file_index}/{total_files}] ✅ {display_name} completed",
        )
        return True
    except Exception as exc:
        status_text.error(f"{display_name} ingestion failed: {exc}")
        return False
    finally:
        TraceCollector().collect(trace)


def _ingest_uploaded_file(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    params: Dict[str, Any],
    file_index: int,
    total_files: int,
) -> bool:
    """Save one uploaded file to temp path and run ingestion pipeline."""
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        return _ingest_file_core(
            tmp_path,
            Path(uploaded_file.name).name,
            collection,
            progress_bar,
            status_text,
            params,
            file_index,
            total_files,
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _run_batch_local_paths(
    paths: List[str],
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    params: Dict[str, Any],
) -> None:
    """Ingest a list of absolute file paths (local batch)."""
    if not paths:
        status_text.warning("No files to ingest.")
        return
    total = len(paths)
    success_count = 0
    for idx, pstr in enumerate(paths, start=1):
        p = Path(pstr)
        ok = _ingest_file_core(
            str(p.resolve()),
            p.name,
            collection,
            progress_bar,
            status_text,
            params,
            file_index=idx,
            total_files=total,
        )
        if ok:
            success_count += 1
    progress_bar.progress(1.0, text=f"✅ Local batch complete: {success_count}/{total} succeeded")
    if success_count == total:
        status_text.success(
            f"Successfully ingested **{total}** local file(s) into **{collection}**."
        )
    else:
        status_text.warning(
            f"Local batch finished with partial failures: {success_count}/{total} succeeded."
        )


def _run_batch_ingestion(
    uploaded_files: list["st.runtime.uploaded_file_manager.UploadedFile"],
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
    params: Dict[str, Any],
    max_files: int | None = None,
) -> None:
    """Run ingestion for multiple uploaded files in one click."""
    selected = list(uploaded_files)
    if max_files is not None:
        selected = selected[:max_files]

    if not selected:
        status_text.warning("No files selected.")
        return

    total = len(selected)
    success_count = 0

    for idx, uploaded_file in enumerate(selected, start=1):
        ok = _ingest_uploaded_file(
            uploaded_file=uploaded_file,
            collection=collection,
            progress_bar=progress_bar,
            status_text=status_text,
            params=params,
            file_index=idx,
            total_files=total,
        )
        if ok:
            success_count += 1

    progress_bar.progress(1.0, text=f"✅ Batch complete: {success_count}/{total} succeeded")
    if success_count == total:
        status_text.success(
            f"Successfully ingested **{total}** file(s) into collection **{collection}**."
        )
    else:
        status_text.warning(
            f"Batch finished with partial failures: {success_count}/{total} succeeded."
        )


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 Ingestion Manager")

    from src.core.settings import load_settings, resolve_path
    from src.libs.splitter.splitter_factory import SplitterFactory

    settings = load_settings()

    data_svc: DataService | None = None
    try:
        data_svc = DataService()
    except Exception as exc:
        st.warning(f"Could not connect to vector store for collection list: {exc}")

    chroma_names = _chroma_collection_names(data_svc)
    cfg_collection = (settings.vector_store.collection_name or "default").strip()
    coll_options = _ingest_collection_select_options(chroma_names, cfg_collection)
    default_pick_idx = 0
    if cfg_collection in coll_options:
        default_pick_idx = coll_options.index(cfg_collection)

    # ── Upload section ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    uploaded = st.file_uploader(
        "Select file(s) to ingest",
        type=["pdf"],
        accept_multiple_files=True,
        key="ingest_uploader",
    )

    mg1, mg2, mg3 = st.columns(3)
    with mg1:
        grade_sel = st.selectbox(
            "年级 (grade) *",
            options=["7", "8", "9"],
            index=0,
            key="ingest_grade",
            help="与 CLI `--grade` / `-g` 一致，写入 Document / Chunk metadata。",
        )
    with mg2:
        volume_sel = st.selectbox(
            "册次 (volume) *",
            options=["上册", "下册"],
            index=0,
            key="ingest_volume",
            help="与 CLI `--volume` / `-V` 一致，写入 Document / Chunk metadata。",
        )
    with mg3:
        pick = st.selectbox(
            "目标集合 (collection) *",
            options=coll_options,
            index=min(default_pick_idx, len(coll_options) - 1),
            key="ingest_collection_pick",
            help=(
                "列表来自当前 Chroma 持久化目录中的集合；默认选中 settings.yaml 的 "
                f"`vector_store.collection_name`（当前 **{cfg_collection}**）。"
                "选最后一项可输入全新集合名。"
            ),
        )
        if pick == _NEW_COLLECTION_OPTION:
            collection = st.text_input(
                "新集合名 *",
                value="",
                placeholder="e.g. math_g8_v1",
                key="ingest_collection_custom",
                help="与 CLI `--collection` / `-c` 一致；将创建新 Chroma 集合并写入 chunk metadata。",
            )
        else:
            collection = pick

    _collection_tier = _locked_chunk_params_for_collection((collection or "").strip())

    st.caption(
        f"Chroma 中已有 **{len(chroma_names)}** 个集合；"
        "导入到已有集合可与 CLI / Ask 使用同一 collection。"
    )
    if _collection_tier:
        st.caption(
            f"📌 **Tier lock:** 当前 collection 名匹配 `_c500` / `_c1000` / `_c1500` → 自动使用 "
            f"Chunk **{_collection_tier[0]}** + Overlap **{_collection_tier[1]}**；"
            "若 Advanced 中选择 **hierarchical**，则仍固定为 **1000 / 200**。"
        )
    ingest_cfg = settings.ingestion
    if ingest_cfg is None:
        st.error("Missing `ingestion` config in settings.")
        return

    splitter_options = SplitterFactory.list_providers() or ["recursive"]
    current_splitter = str(ingest_cfg.splitter)
    if current_splitter not in splitter_options:
        splitter_options.append(current_splitter)

    with st.expander("⚙️ Advanced Ingestion Parameters (affects retrieval quality)", expanded=False):
        st.caption("这些参数会影响切片和索引质量；改动后仅对本次新导入生效。")
        splitter = st.selectbox(
            "Splitter",
            options=splitter_options,
            index=splitter_options.index(current_splitter),
            key="ingest_splitter_select",
            help="Available splitters are discovered from SplitterFactory registrations.",
        )
        hierarchical_mode = str(splitter) == "hierarchical"
        locked_pair = (
            None
            if hierarchical_mode
            else _locked_chunk_params_for_collection((collection or "").strip())
        )
        if locked_pair:
            st.caption(
                f"目标集合 **`{(collection or '').strip()}`** 匹配教材实验命名（`_c500` / `_c1000` / `_c1500`）→ "
                f"已锁定 **Chunk Size = {locked_pair[0]}**，**Chunk Overlap = {locked_pair[1]}**。"
            )
        p1, p2, p3 = st.columns(3)
        with p1:
            _size_default, _ov_default = (
                locked_pair
                if locked_pair
                else (int(ingest_cfg.chunk_size), int(ingest_cfg.chunk_overlap))
            )
            chunk_size_input = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=_size_default,
                step=50,
                disabled=hierarchical_mode or locked_pair is not None,
            )
            chunk_overlap_input = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=_ov_default,
                step=20,
                disabled=hierarchical_mode or locked_pair is not None,
            )
            batch_size = st.number_input(
                "Batch Size (Embedding)",
                min_value=1,
                max_value=128,
                value=int(ingest_cfg.batch_size),
                step=1,
            )
        with p2:
            transform_llm_enabled = st.checkbox(
                "Enable LLM Transform (总开关)",
                value=bool((ingest_cfg.chunk_refiner or {}).get("use_llm", False))
                or bool((ingest_cfg.metadata_enricher or {}).get("use_llm", False)),
                help="统一控制 transform 阶段 LLM：ChunkRefiner + MetadataEnricher",
            )
            image_caption_enabled = st.checkbox(
                "Enable Image Captioning (Vision LLM)",
                value=bool(settings.vision_llm.enabled) if settings.vision_llm else False,
            )
        with p3:
            chunk_refiner_use_llm = st.checkbox(
                "Chunk Refiner use_llm",
                value=bool((ingest_cfg.chunk_refiner or {}).get("use_llm", False)),
                disabled=not transform_llm_enabled,
            )
            metadata_enricher_use_llm = st.checkbox(
                "Metadata Enricher use_llm",
                value=bool((ingest_cfg.metadata_enricher or {}).get("use_llm", False)),
                disabled=not transform_llm_enabled,
            )
            force_reingest = st.checkbox(
                "Force Re-ingest (ignore hash dedup)",
                value=False,
                help="重新处理同一份文件内容。未勾选时，已入库文件会被跳过。",
            )
            st.caption("Transform 耗时主要来自两类 LLM 调用：文本精修 + 元数据增强。")
        if hierarchical_mode:
            st.info(
                f"`hierarchical` 模式下，当前固定 `Chunk Size={HIERARCHICAL_FIXED_CHUNK_SIZE}`、"
                f"`Chunk Overlap={HIERARCHICAL_FIXED_CHUNK_OVERLAP}`，用于单独验证层级切分。"
            )

    effective_chunk_size = (
        HIERARCHICAL_FIXED_CHUNK_SIZE
        if hierarchical_mode
        else (locked_pair[0] if locked_pair else int(chunk_size_input))
    )
    effective_chunk_overlap = (
        HIERARCHICAL_FIXED_CHUNK_OVERLAP
        if hierarchical_mode
        else (locked_pair[1] if locked_pair else int(chunk_overlap_input))
    )

    collection_stripped = (collection or "").strip()
    ingest_document_metadata = {
        "grade": str(grade_sel),
        "volume": str(volume_sel),
        "collection": collection_stripped,
    }

    runtime_params = {
        "chunk_size": effective_chunk_size,
        "chunk_overlap": effective_chunk_overlap,
        "splitter": str(splitter),
        "batch_size": int(batch_size),
        "transform_llm_enabled": bool(transform_llm_enabled),
        "chunk_refiner_use_llm": bool(chunk_refiner_use_llm),
        "metadata_enricher_use_llm": bool(metadata_enricher_use_llm),
        "image_caption_enabled": bool(image_caption_enabled),
        "force_reingest": bool(force_reingest),
        "ingest_document_metadata": ingest_document_metadata,
    }

    # ── Local path batch (same runtime_params as upload) ───────────
    st.subheader("📁 Local path batch import")
    st.caption(
        "输入**文件或目录**路径（绝对路径，或相对项目根）；当前仅支持扩展名 pdf。"
        "与上方 **年级 / 册次 / collection** 及 **Advanced** 参数一致。先 **Scan** 再 **Ingest**。"
    )
    from src.observability.dashboard.services.local_ingest_paths import discover_ingest_files

    path_raw = st.text_input(
        "Local path",
        value="",
        key="ingest_local_path_input",
        placeholder=r".\data\your_pdfs 或 C:\books\pdf",
    )
    loc_recursive = st.checkbox("Recursive directory scan", value=True, key="ingest_local_recursive")
    loc_cap = st.number_input(
        "Max files this run",
        min_value=1,
        max_value=500,
        value=60,
        step=1,
        key="ingest_local_max_files",
    )
    lc1, lc2 = st.columns(2)
    with lc1:
        scan_local = st.button("🔎 Scan path", key="ingest_local_scan_btn")
    with lc2:
        run_local = st.button("🚀 Ingest scanned files", key="ingest_local_ingest_btn")

    if scan_local:
        raw = (path_raw or "").strip()
        if not raw:
            st.warning("请输入路径。")
        else:
            try:
                root = Path(raw)
                if not root.is_absolute():
                    root = resolve_path(root)
                root = root.resolve()
                found = discover_ingest_files(root, recursive=bool(loc_recursive))
                st.session_state["ingest_local_paths_list"] = [str(x.resolve()) for x in found]
                st.session_state["ingest_local_scan_label"] = str(root)
            except Exception as exc:
                st.session_state.pop("ingest_local_paths_list", None)
                st.error(f"Scan failed: {exc}")

    scanned_paths = st.session_state.get("ingest_local_paths_list")
    if scanned_paths:
        lbl = st.session_state.get("ingest_local_scan_label", "")
        st.caption(f"已扫描 **{len(scanned_paths)}** 个文件 · `{lbl}`")
        preview = [{"path": p} for p in scanned_paths[:50]]
        st.dataframe(preview, use_container_width=True, height=min(360, 80 + len(preview) * 28))
        if len(scanned_paths) > 50:
            st.caption(f"… 仅展示前 50 条，共 {len(scanned_paths)} 条将参与导入（受 Max files 限制）。")

    if run_local:
        if not collection_stripped:
            st.error("请选择或填写目标 collection。")
        elif not scanned_paths:
            st.warning("请先 **Scan path** 成功后再导入。")
        else:
            take = scanned_paths[: int(loc_cap)]
            progress_bar = st.progress(0, text="Preparing local batch ingestion…")
            status_text = st.empty()
            _run_batch_local_paths(
                paths=take,
                collection=collection_stripped,
                progress_bar=progress_bar,
                status_text=status_text,
                params=runtime_params,
            )

    selected_count = len(uploaded) if uploaded else 0
    if selected_count > 0:
        st.caption(f"Selected files: {selected_count}")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_all = st.button("🚀 Start Ingestion (All selected)", key="btn_ingest_all")
        with btn_col2:
            start_five = st.button("⚡ One-click Ingest First 5", key="btn_ingest_5")

        if start_all or start_five:
            if not collection_stripped:
                st.error(
                    "请选择上方「目标集合」中的已有集合，或选「➕ New collection…」后在「新集合名」中填写名称 "
                    "（与 CLI `--collection` / `-c` 一致）。"
                )
            else:
                progress_bar = st.progress(0, text="Preparing batch ingestion…")
                status_text = st.empty()
                _run_batch_ingestion(
                    uploaded_files=uploaded,
                    collection=collection_stripped,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    params=runtime_params,
                    max_files=5 if start_five else None,
                )

    st.divider()

    # ── Document management section ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    coll_for_list = collection_stripped or "default"

    list_svc = data_svc
    if list_svc is None:
        try:
            list_svc = DataService()
        except Exception as exc:
            st.error(f"Failed to load documents: {exc}")
            return

    try:
        docs = list_svc.list_documents(coll_for_list)
        history_rows = list_svc.list_ingestion_history(coll_for_list, limit=40)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if history_rows:
        with st.expander(
            "📜 Recent ingestion activity (this collection)",
            expanded=(len(docs) == 0),
        ):
            st.caption(
                "Shows SQLite records for this collection (success + failed). "
                "Dashboard uploads use a temp path; the **file hash** still matches. "
                "Details: **Ingestion Traces**."
            )
            st.table(
                [
                    {
                        "Status": r.get("status", "—"),
                        "Updated": (r.get("updated_at") or "—")[:19],
                        "File": _short_path(r.get("file_path", "")),
                        "Error preview": (r.get("error_msg") or "")[:160],
                    }
                    for r in history_rows
                ]
            )

    if not docs:
        st.info(
            "**No successful ingests in this collection yet** "
            "(vectors were not stored or ingest failed). "
            "See **Recent ingestion activity** above or **Ingestion Traces**. "
            "Upload a file and click **Start Ingestion**, or enable **Force Re-ingest**."
        )
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"collection: `{doc.get('collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_btn:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    result = list_svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection") or coll_for_list,
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"Deleted: {result.chunks_deleted} chunks, "
                            f"{result.images_deleted} images removed."
                        )
                        st.rerun()
                    else:
                        st.warning(f"Partial delete. Errors: {result.errors}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
