"""Data Browser page – browse ingested documents, chunks, and images.

Document list is built from **Chroma chunk metadatas** (not SQLite), so
collections with vectors but no ``ingestion_history`` row still appear.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.core.settings import load_settings
from src.observability.dashboard.services.data_service import DataService

_CHUNK_PREVIEW_LIMIT = 80
# Inline thumbnails per document — avoids huge DOM + one bad image killing the page.
_MAX_IMAGES_INLINE_DEFAULT = 48


def _safe_st_image(img_path: Path, caption: str, *, width: int = 200) -> None:
    """Render ``st.image`` only if the file is readable and has positive dimensions."""
    if not img_path.is_file():
        st.caption(f"{caption} (file missing)")
        return
    try:
        if img_path.stat().st_size <= 0:
            st.caption(f"{caption} (empty file)")
            return
    except OSError:
        st.caption(f"{caption} (unreadable path)")
        return

    try:
        from PIL import Image

        with Image.open(img_path) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                st.caption(f"{caption} (invalid size {w}×{h})")
                return
    except ImportError:
        pass  # Pillow optional; fall through and let st.image try
    except Exception as exc:
        st.caption(f"{caption} (cannot decode: {type(exc).__name__})")
        return

    try:
        st.image(str(img_path), caption=caption, width=width)
    except Exception as exc:
        st.caption(f"{caption} (display failed: {type(exc).__name__}: {exc})")


def _short_path(path_str: str, *, max_name: int = 72) -> str:
    """Show basename only; temp paths are still readable in history tables."""
    if not path_str:
        return "—"
    name = Path(path_str).name
    if len(name) <= max_name:
        return name
    return name[: max_name - 1] + "…"


def _ordered_collections(names: list[str], preferred: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    pref = (preferred or "default").strip() or "default"
    for n in [pref] + sorted(names):
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out if out else ["default"]


def render() -> None:
    """Render the Data Browser page."""
    st.header("🔍 Data Browser")
    st.caption(
        "文档列表由 **Chroma 中该 collection 的 chunk 元数据** 聚合得到；"
        "与 SQLite 入库记录无关，因此能看到仅有向量、无 integrity 记录的集合。"
    )

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    settings = load_settings()
    cfg_col = (settings.vector_store.collection_name or "default").strip()

    collections = svc.list_collections() or []
    collections = _ordered_collections(collections, cfg_col)

    collection = st.selectbox(
        "Collection",
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    try:
        total_chroma = svc.chroma_chunk_count(coll_arg)
        docs = svc.list_documents_from_chroma(coll_arg)
    except Exception as exc:
        st.error(f"Failed to load from Chroma: {exc}")
        return

    # Optional filters (metadata on chunks)
    all_grades: set[str] = set()
    all_ct: set[str] = set()
    for d in docs:
        all_grades.update(d.get("grades") or [])
        all_ct.update(d.get("content_types") or [])
    f1, f2 = st.columns(2)
    with f1:
        grade_filter = st.multiselect(
            "Filter by grade (metadata)",
            options=sorted(all_grades) if all_grades else [],
            default=[],
            key="db_grade_filter",
        )
    with f2:
        ct_filter = st.multiselect(
            "Filter by content_type",
            options=sorted(all_ct) if all_ct else [],
            default=[],
            key="db_ct_filter",
        )

    def _doc_matches_filters(doc: dict) -> bool:
        gset = set(doc.get("grades") or [])
        cset = set(doc.get("content_types") or [])
        if grade_filter and not (gset & set(grade_filter)):
            return False
        if ct_filter and not (cset & set(ct_filter)):
            return False
        return True

    docs = [d for d in docs if _doc_matches_filters(d)]

    scanned_total = sum(int(d.get("chunk_count", 0)) for d in docs)
    scan_meta = getattr(svc, "_last_chroma_scan", None) or {}
    if scan_meta.get("capped"):
        st.warning(
            "Chroma 元数据扫描已达到 **上限**，列表可能不完整。"
            "可在代码中提高 ``DataService.list_documents_from_chroma(..., max_chunks_to_scan=...)``。"
        )

    # ── Danger zone: clear all data ────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.warning(
            "This will **permanently delete** all data: "
            "ChromaDB collections, BM25 indexes, images, ingestion history, and trace logs."
        )
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button("🗑️ Clear All Data", type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("Are you sure? This action cannot be undone!")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ Yes, delete everything", key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"Cleared with {len(result['errors'])} error(s): "
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"All data cleared! "
                            f"{result['collections_deleted']} collection(s) deleted."
                        )
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    if not docs and total_chroma == 0:
        st.info(
            "**该 collection 在 Chroma 中暂无向量。** "
            "请在 **Ingestion Manager** 导入文档，或切换到其他 collection。"
        )
        return

    if not docs:
        st.info(
            "**没有符合当前筛选条件的文档。** "
            "请清空年级 / content_type 筛选，或确认该 collection 的 chunk 是否带有对应 metadata。"
        )
        return

    st.subheader(f"📄 Documents ({len(docs)}) · {scanned_total} chunks in view")

    show_all = st.checkbox(
        "Show all chunks per document (may be slow)",
        value=False,
        key="db_show_all_chunks",
    )
    chunk_limit = None if show_all else _CHUNK_PREVIEW_LIMIT

    for idx, doc in enumerate(docs):
        source_name = Path(str(doc.get("source_path", ""))).name
        n_chunks = int(doc.get("chunk_count", 0))
        label = (
            f"📑 {source_name}  —  {n_chunks} chunks · "
            f"{doc.get('image_count', 0)} images"
        )
        grades = doc.get("grades") or []
        cts = doc.get("content_types") or []
        if grades or cts:
            label += f" · grades:{','.join(grades) or '—'} · types:{','.join(cts) or '—'}"

        with st.expander(label, expanded=(len(docs) == 1)):
            action_col, _ = st.columns([1, 4])
            with action_col:
                if st.button("🗑️ Delete This Document", key=f"db_del_doc_{idx}"):
                    try:
                        doc_collection = doc.get("collection") or collection or "default"
                        result = svc.delete_document(
                            source_path=str(doc.get("source_path", "")),
                            collection=doc_collection,
                            source_hash=doc.get("source_hash") or None,
                        )
                        if getattr(result, "success", False):
                            st.success(
                                f"Deleted {source_name}: "
                                f"{getattr(result, 'chunks_deleted', 0)} chunks, "
                                f"{getattr(result, 'images_deleted', 0)} images."
                            )
                        else:
                            st.warning(
                                f"Delete finished with issues: {getattr(result, 'errors', [])}"
                            )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

            st.divider()

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunks", n_chunks)
            col_b.metric("Images", doc.get("image_count", 0))
            col_c.metric("Collection", doc.get("collection", "—"))
            sh = doc.get("source_hash") or ""
            hash_disp = f"`{sh[:24]}…`" if len(sh) > 24 else (f"`{sh}`" if sh else "`—`")
            st.caption(
                f"**Source:** {doc.get('source_path', '—')}  ·  **doc_hash:** {hash_disp}  ·  "
                f"**SQLite processed_at:** {doc.get('processed_at') or '— (Chroma-only)'}",
            )

            st.divider()

            chunks = svc.get_chunks(
                str(sh),
                coll_arg,
                limit=chunk_limit,
                source_path=None if sh else str(doc.get("source_path", "")),
            )
            if chunks and chunk_limit and n_chunks > chunk_limit:
                st.caption(
                    f"Showing first **{len(chunks)}** of **{n_chunks}** chunks. "
                    "勾选上方 **Show all chunks** 可查看全部（可能卡顿）。"
                )
            if chunks:
                st.markdown(f"### 📦 Chunks ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {}) or {}
                    chunk_id = chunk["id"]

                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            f"**Chunk {cidx + 1}** · `{str(chunk_id)[-24:]}` · "
                            f"{len(text)} chars"
                        )
                        _height = max(120, min(len(text) // 2, 400))
                        st.text_area(
                            "Content",
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        with st.expander("📋 Metadata", expanded=False):
                            st.json(meta)
            else:
                st.caption("No chunks returned for this document (check doc_hash / source_path in Chroma).")

            sh_img = str(doc.get("source_hash") or "")
            if sh_img:
                images = svc.get_images(sh_img, coll_arg)
                if images:
                    st.divider()
                    st.markdown(f"### 🖼️ Images ({len(images)})")
                    show_all_img = st.checkbox(
                        "Show all images inline (very slow for large books)",
                        value=False,
                        key=f"db_show_all_img_{idx}",
                    )
                    cap = len(images) if show_all_img else min(len(images), _MAX_IMAGES_INLINE_DEFAULT)
                    preview = images[:cap]
                    if len(images) > cap:
                        st.caption(
                            f"预览前 **{cap}** / **{len(images)}** 张；勾选上方可展开全部（仍跳过损坏/零尺寸图）。"
                        )
                    img_cols = st.columns(min(len(preview), 4))
                    for iidx, img in enumerate(preview):
                        with img_cols[iidx % len(img_cols)]:
                            img_path = Path(str(img.get("file_path", "")))
                            _safe_st_image(img_path, str(img.get("image_id", "image")))
