"""Ingestion Traces page – browse ingestion trace history with per-stage detail.

Layout:
1. Trace list (reverse-chronological, filtered to trace_type=="ingestion")
2. Pipeline overview: source file, total time, stage timing waterfall
3. Per-stage detail tabs:
   📄 Load    – raw document text preview
   ✂️ Split   – chunk list with text
   🔄 Transform – before/after diff, enrichment metadata
   🔢 Embed   – vector stats
   💾 Upsert  – stored IDs
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.pages._trace_ui import (
    TRACE_PAGE_SIZE,
    UI_JSON_TEXT_CAP,
    paginate_traces,
    render_truncated_text_area,
)
from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)

# Max chars per large text field when embedding chunk rows in JSON code blocks.
_TRANSFORM_CHUNK_JSON_TEXT_CAP = UI_JSON_TEXT_CAP


def _chunk_dict_for_json_code(
    chunk: Dict[str, Any],
    *,
    text_cap: int = _TRANSFORM_CHUNK_JSON_TEXT_CAP,
) -> Dict[str, Any]:
    """Copy chunk row; truncate huge text fields so Streamlit stays responsive."""
    out = dict(chunk)
    for key in ("text_before", "text_after", "text"):
        val = out.get(key)
        if isinstance(val, str) and len(val) > text_cap:
            out[key] = val[:text_cap] + "\n\n... [truncated for trace UI]"
    return out


@st.cache_data(ttl=30, show_spinner=False)
def _load_traces_cached():
    """Cached trace loader – avoids re-parsing traces.jsonl every render."""
    svc = TraceService()
    return svc.list_traces(trace_type="ingestion")


def render() -> None:
    """Render the Ingestion Traces page."""
    st.header("🔬 Ingestion Traces")
    st.caption("只读查看 `logs/traces.jsonl` 中的入库链路记录，不触发重新入库或评测。")

    svc = TraceService()
    traces = _load_traces_cached()

    if not traces:
        st.info("No ingestion traces recorded yet. Run an ingestion first!")
        return

    st.subheader(f"📋 Trace History ({len(traces)})")
    page_traces, page_base, _ = paginate_traces(traces, page_key="ingestion_trace_page")

    for local_idx, trace in enumerate(page_traces):
        trace_idx = page_base * TRACE_PAGE_SIZE + local_idx
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        source_path = meta.get("source_path", "—")

        # Build expander title
        file_name = source_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if source_path != "—" else "—"
        expander_title = f"📄 **{file_name}** · {total_label} · {started[:19]}"

        with st.expander(expander_title, expanded=False):
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            # ── 1. Overview metrics ────────────────────────────
            st.markdown("#### 📊 Pipeline Overview")
            st.caption(f"Source: `{source_path}`")

            load_d = stages_by_name.get("load", {}).get("data", {})
            split_d = stages_by_name.get("split", {}).get("data", {})
            transform_d = stages_by_name.get("transform", {}).get("data", {})
            embed_d = stages_by_name.get("embed", {}).get("data", {})
            upsert_d = stages_by_name.get("upsert", {}).get("data", {})

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Doc Length", f"{load_d.get('text_length', 0):,} chars")
            with c2:
                st.metric("Chunks", split_d.get("chunk_count", 0))
            with c3:
                st.metric("Images", load_d.get("image_count", 0))
            with c4:
                st.metric("Vectors", upsert_d.get("vector_count", 0))
            with c5:
                st.metric("Total Time", total_label)

            st.divider()

            # ── 2. Stage timing waterfall ──────────────────────
            # Filter to main pipeline stages only (not sub-stages)
            main_stages = [
                t for t in timings
                if t["stage_name"] in ("load", "split", "transform", "embed", "upsert")
            ]
            if main_stages:
                st.markdown("#### ⏱️ Stage Timings")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_stages}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "Stage": t["stage_name"],
                        "Elapsed (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_stages
                ])

            # ── Diagnostics ───────────────────────────────────
            _render_ingestion_diagnostics(stages_by_name, load_d, split_d, transform_d, embed_d, upsert_d)

            st.divider()

            # ── 3. Per-stage detail tabs ───────────────────────
            st.markdown("#### 🔍 Stage Details")

            tab_defs = []
            if "load" in stages_by_name:
                tab_defs.append(("📄 Load", "load"))
            if "split" in stages_by_name:
                tab_defs.append(("✂️ Split", "split"))
            if "transform" in stages_by_name:
                tab_defs.append(("🔄 Transform", "transform"))
            if "embed" in stages_by_name:
                tab_defs.append(("🔢 Embed", "embed"))
            if "upsert" in stages_by_name:
                tab_defs.append(("💾 Upsert", "upsert"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "load":
                            _render_load_stage(data, trace_idx=trace_idx)
                        elif key == "split":
                            _render_split_stage(data, trace_idx=trace_idx)
                        elif key == "transform":
                            _render_transform_stage(data, trace_idx=trace_idx)
                        elif key == "embed":
                            _render_embed_stage(
                                data,
                                transform_data=stages_by_name.get(
                                    "transform", {}
                                ).get("data"),
                                trace_idx=trace_idx,
                            )
                        elif key == "upsert":
                            _render_upsert_stage(data)
            else:
                st.info("No stage details available.")


def _render_ingestion_diagnostics(
    stages_by_name: Dict[str, Any],
    load_d: Dict[str, Any],
    split_d: Dict[str, Any],
    transform_d: Dict[str, Any],
    embed_d: Dict[str, Any],
    upsert_d: Dict[str, Any],
) -> None:
    """Render diagnostic hints for ingestion pipeline stages."""
    expected = ["load", "split", "transform", "embed", "upsert"]
    present = [s for s in expected if s in stages_by_name]
    missing = [s for s in expected if s not in stages_by_name]

    if missing:
        missing_labels = {"load": "📄 Load", "split": "✂️ Split", "transform": "🔄 Transform", "embed": "🔢 Embed", "upsert": "💾 Upsert"}
        names = ", ".join(missing_labels.get(m, m) for m in missing)
        integrity_d = stages_by_name.get("integrity", {}).get("data", {})
        if "load" in missing:
            if integrity_d.get("skipped") and integrity_d.get("reason") == "already_processed":
                st.info(
                    f"**Pipeline skipped by dedup — missing stages: {names}.** "
                    "This file hash was already ingested. Re-run with `Force Re-ingest` to process again."
                )
            else:
                st.error(
                    f"**Pipeline incomplete — missing stages: {names}.** "
                    "The Load stage failed or was skipped. The document may be corrupted or unsupported."
                )
        else:
            st.warning(
                f"**Pipeline incomplete — missing stages: {names}.** "
                "An error may have occurred during processing. Check the logs for details."
            )

    # Stage-specific diagnostics
    if "load" in stages_by_name and load_d.get("text_length", 0) == 0:
        st.warning("**Load stage produced empty text.** The document may be image-only or in an unsupported format.")

    if "split" in stages_by_name and split_d.get("chunk_count", 0) == 0:
        st.warning("**Split stage produced 0 chunks.** The document text may be too short or empty.")

    if "transform" in stages_by_name:
        refined_llm = transform_d.get("refined_by_llm", 0)
        refined_rule = transform_d.get("refined_by_rule", 0)
        if refined_llm == 0 and refined_rule == 0:
            st.info("Transform: No chunks were refined. LLM refinement may be disabled or skipped for short chunks.")
        drop_n = int(transform_d.get("chunks_dropped_before_embed") or 0)
        if drop_n > 0:
            st.info(
                f"Transform: {drop_n} chunk(s) were removed before embedding because the "
                "post-transform text was empty or shorter than 10 characters. "
                "They still appear in the Transform tab for inspection."
            )
        r_err = int(transform_d.get("refined_by_error") or 0)
        e_err = int(transform_d.get("enriched_by_error") or 0)
        if r_err or e_err:
            st.warning(
                f"Transform stage reported errors on {r_err} refine / {e_err} enrich chunk(s). "
                "Expand per-chunk JSON or check logs."
            )

    if "embed" in stages_by_name and embed_d.get("dense_vector_count", 0) == 0:
        st.warning("**Embed stage produced 0 vectors.** Embedding API may have failed. Check API key and endpoint.")

    if "upsert" in stages_by_name:
        vec_count = upsert_d.get("vector_count", upsert_d.get("dense_store", {}).get("count", 0))
        if vec_count == 0:
            st.warning("**Upsert stage stored 0 vectors.** Database write may have failed.")

    # Check for error fields in any stage data
    for stage_name in present:
        stage_data = stages_by_name[stage_name].get("data", {})
        err = stage_data.get("error", "")
        if err:
            label = stage_name.replace("_", " ").title()
            st.error(f"**{label} stage error:** {err}")


# ═══════════════════════════════════════════════════════════════
# Per-stage renderers
# ═══════════════════════════════════════════════════════════════

def _render_load_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Load stage: raw document preview."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Doc ID", data.get("doc_id", "—")[:16])
    with c2:
        st.metric("Text Length", f"{data.get('text_length', 0):,}")
    with c3:
        st.metric("Images", data.get("image_count", 0))

    preview = data.get("text_preview", "")
    if preview:
        st.markdown("**Raw Document Text**")
        render_truncated_text_area(
            preview,
            key=f"load_raw_text_{trace_idx}",
            max_height=500,
        )
    else:
        st.info("No text preview recorded in this trace.")


def _render_split_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Split stage: chunk list with texts."""
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chunks", data.get("chunk_count", 0))
    with c2:
        st.metric("Avg Size", f"{data.get('avg_chunk_size', 0)} chars")

    chunks = data.get("chunks", [])
    if chunks:
        show_n = min(len(chunks), 15)
        st.markdown(f"**Chunks after splitting** (showing {show_n}/{len(chunks)})")
        for i, chunk in enumerate(chunks[:show_n]):
            char_len = chunk.get("char_len", 0)
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            header = f"📝 **Chunk #{i+1}** — `{chunk_id[:20]}` — {char_len} chars"
            with st.expander(header, expanded=False):
                render_truncated_text_area(
                    text,
                    key=f"split_{trace_idx}_{i}",
                    max_height=350,
                )
    else:
        st.info("No chunk text recorded. Re-run ingestion to generate new traces.")


def _render_transform_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Transform stage: before/after refinement + enrichment metadata."""
    method = data.get("method") or "—"
    st.caption(f"**Pipeline:** `{method}`")
    if data.get("chunk_refiner_use_llm") is not None or data.get("metadata_enricher_use_llm") is not None:
        cr = data.get("chunk_refiner_use_llm")
        me = data.get("metadata_enricher_use_llm")
        ic = data.get("image_caption_enabled")
        st.caption(
            f"Refiner LLM: `{cr}` · Enricher LLM: `{me}` · Image caption: `{ic}`"
        )

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Refined (LLM / Rule)",
            f"{data.get('refined_by_llm', 0)} / {data.get('refined_by_rule', 0)}",
        )
    with c2:
        st.metric(
            "Enriched (LLM / Rule)",
            f"{data.get('enriched_by_llm', 0)} / {data.get('enriched_by_rule', 0)}",
        )
    with c3:
        st.metric("Captioned", data.get("captioned_chunks", 0))

    r_err = int(data.get("refined_by_error") or 0)
    e_err = int(data.get("enriched_by_error") or 0)
    if r_err or e_err:
        c4, c5 = st.columns(2)
        with c4:
            st.metric("Refine errors", r_err)
        with c5:
            st.metric("Enrich errors", e_err)

    n_after = data.get("chunks_after_transform")
    n_drop = data.get("chunks_dropped_before_embed")
    n_kept = data.get("chunks_kept_for_embed")
    if n_after is not None or n_drop is not None or n_kept is not None:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("After transform", int(n_after) if n_after is not None else 0)
        with d2:
            st.metric("Dropped (<10 chars)", int(n_drop) if n_drop is not None else 0)
        with d3:
            st.metric("Kept for embed", int(n_kept) if n_kept is not None else 0)

    chunks = data.get("chunks", [])
    if chunks:
        st.markdown("**Per-chunk transform results**")
        captioned_indices = [
            i
            for i, chunk in enumerate(chunks)
            if "Description:" in (chunk.get("text_after") or "")
            or int(chunk.get("image_caption_count") or 0) > 0
        ]
        if captioned_indices:
            first_pos = captioned_indices[0] + 1
            last_pos = captioned_indices[-1] + 1
            st.caption(
                f"Detected image descriptions in {len(captioned_indices)} chunks "
                f"(positions #{first_pos} ~ #{last_pos})."
            )
        else:
            st.caption("No `(Description: ...)` text found in transformed chunks.")

        show_captioned_only = st.checkbox(
            "Show only chunks with image descriptions",
            value=False,
            key=f"transform_caption_only_{trace_idx}",
        )
        iter_items = (
            [(i, chunks[i]) for i in captioned_indices]
            if show_captioned_only
            else list(enumerate(chunks))
        )
        max_transform_rows = 20
        if len(iter_items) > max_transform_rows:
            st.caption(
                f"Showing first {max_transform_rows} of {len(iter_items)} chunks "
                "(enable filter or open trace JSON on disk for full data)."
            )
            iter_items = iter_items[:max_transform_rows]

        for i, chunk in iter_items:
            chunk_id = chunk.get("chunk_id", "")
            refined_by = chunk.get("refined_by", "")
            enriched_by = chunk.get("enriched_by", "")
            text_before = chunk.get("text_before", "")
            text_after = chunk.get("text_after", "")
            desc_count = text_after.count("Description:") if text_after else 0
            kept = chunk.get("kept_for_embedding")
            if kept is None:
                stripped = (text_after or "").strip()
                kept = bool(stripped) and len(stripped) >= 10

            badge_parts = []
            if refined_by:
                badge_parts.append(f"refined:`{refined_by}`")
            if enriched_by:
                badge_parts.append(f"enriched:`{enriched_by}`")
            if desc_count:
                badge_parts.append(f"image_desc:`{desc_count}`")
            ipc = chunk.get("image_placeholder_count")
            if ipc is not None and int(ipc) > 0:
                badge_parts.append(f"[IMAGE]:`{ipc}`")
            icc = chunk.get("image_caption_count")
            if icc is not None and int(icc) > 0:
                badge_parts.append(f"img_caps:`{icc}`")
            chn = chunk.get("chapter_num")
            secn = chunk.get("section_num")
            if chn is not None or secn is not None:
                badge_parts.append(f"ch/sec:`{chn}`/`{secn}`")
            if chunk.get("refine_fallback_reason"):
                badge_parts.append(f"ref_fb:`{chunk['refine_fallback_reason']}`")
            if chunk.get("enrich_fallback_reason"):
                badge_parts.append(f"enr_fb:`{chunk['enrich_fallback_reason']}`")
            if chunk.get("enrich_error"):
                badge_parts.append("enrich_err:yes")
            if kept is False:
                badge_parts.append("not_embedded")
            badges = " · ".join(badge_parts)

            idx = chunk.get("chunk_index")
            page = chunk.get("page")
            pos = f"idx={idx}" if idx is not None else ""
            if page is not None:
                pos = f"{pos} · p.{page}" if pos else f"p.{page}"
            pos_suffix = f" — {pos}" if pos else ""

            header = f"🔄 **Chunk #{i+1}** — `{chunk_id[:20]}`{pos_suffix} — {badges}"
            with st.expander(header, expanded=False):
                if chunk.get("enrich_error"):
                    st.error(str(chunk["enrich_error"]))
                with st.expander("Chunk metadata (JSON)", expanded=False):
                    try:
                        st.code(
                            json.dumps(
                                _chunk_dict_for_json_code(chunk),
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ),
                            language="json",
                        )
                    except Exception as exc:
                        logger.debug("Chunk JSON serialize failed: %s", exc)
                        st.code(str(_chunk_dict_for_json_code(chunk)), language=None)

                if text_before or text_after:
                    st.markdown("**Text Comparison**")
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("*Before refinement:*")
                        render_truncated_text_area(
                            text_before or "(empty)",
                            key=f"transform_before_{trace_idx}_{i}",
                        )
                    with col_after:
                        st.markdown("*After refinement + enrichment:*")
                        render_truncated_text_area(
                            text_after or "(empty)",
                            key=f"transform_after_{trace_idx}_{i}",
                        )
    else:
        st.info("No per-chunk transform data recorded. Re-run ingestion for new traces.")


def _render_embed_stage(
    data: Dict[str, Any],
    *,
    transform_data: Optional[Dict[str, Any]] = None,
    trace_idx: int = 0,
) -> None:
    """Render Embed stage: dual-path Dense + Sparse encoding details."""
    # ── Overview metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dense Vectors", data.get("dense_vector_count", 0))
    with c2:
        st.metric("Dimension", data.get("dense_dimension", 0))
    with c3:
        st.metric("Sparse Docs", data.get("sparse_doc_count", 0))
    with c4:
        st.metric("Method", data.get("method", "—"))

    chunks = data.get("chunks", [])
    if not chunks:
        st.info("No chunk encoding data recorded.")
        return

    meta_by_chunk_id: Dict[str, Dict[str, Any]] = {}
    for row in (transform_data or {}).get("chunks") or []:
        cid = row.get("chunk_id") or ""
        if cid:
            meta_by_chunk_id[cid] = row

    # ── Dual-path per-chunk table ──
    st.markdown("---")
    dense_tab, sparse_tab = st.tabs(["🟦 Dense Encoding", "🟨 Sparse Encoding (BM25)"])

    with dense_tab:
        st.markdown("Each chunk → **float vector** via embedding model (e.g. `text-embedding-ada-002`)")
        dense_rows = []
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            dense_rows.append({
                "#": i + 1,
                "Chunk ID": chunk.get("chunk_id", ""),
                "Chars": char_len,
                "Est. Tokens": max(1, char_len // 3),
                "Dense Dim": chunk.get("dense_dim", data.get("dense_dimension", "—")),
            })
        st.table(dense_rows)

    with sparse_tab:
        st.markdown("Each chunk → **term frequency stats** for BM25 indexing")
        sparse_rows = []
        for i, chunk in enumerate(chunks):
            sparse_rows.append({
                "#": i + 1,
                "Chunk ID": chunk.get("chunk_id", ""),
                "Doc Length (terms)": chunk.get("doc_length", "—"),
                "Unique Terms": chunk.get("unique_terms", "—"),
            })
        st.table(sparse_rows)

        # Top terms per chunk
        for i, chunk in enumerate(chunks):
            top_terms = chunk.get("top_terms", [])
            if top_terms:
                with st.expander(f"🔤 Chunk {i + 1} — Top Terms", expanded=False):
                    term_rows = [{"Term": t["term"], "Freq": t["freq"]} for t in top_terms]
                    st.table(term_rows)

    # ── Per-chunk enrichment metadata (same trace Transform stage) ──
    if meta_by_chunk_id:
        st.markdown("---")
        st.markdown("**Enrichment metadata** (from Transform stage, matched by `chunk_id`)")
        for i, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id", "")
            trow = meta_by_chunk_id.get(cid)
            if not trow:
                continue
            title = trow.get("title") or ""
            tags = trow.get("tags") or []
            summary = trow.get("summary") or ""
            refined_by = trow.get("refined_by") or ""
            enriched_by = trow.get("enriched_by") or ""
            enr_err = trow.get("enrich_error") or ""
            enr_fb = trow.get("enrich_fallback_reason") or ""
            ref_fb = trow.get("refine_fallback_reason") or ""
            if not any([title, tags, summary, refined_by, enriched_by, enr_err, enr_fb, ref_fb]):
                continue
            badges = " · ".join(
                p
                for p in (
                    f"refined:`{refined_by}`" if refined_by else "",
                    f"enriched:`{enriched_by}`" if enriched_by else "",
                    f"ref_fb:`{ref_fb}`" if ref_fb else "",
                    f"enr_fb:`{enr_fb}`" if enr_fb else "",
                )
                if p
            )
            cid_disp = cid[:28] + "…" if len(cid) > 28 else cid
            header = f"📎 Chunk #{i + 1} — `{cid_disp}` — {badges}"
            with st.expander(header, expanded=False):
                if enr_err:
                    st.error(f"Enrich error: {enr_err}")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(f"**Title:** {title}" if title else "_No title_")
                with mc2:
                    if tags:
                        st.markdown("**Tags:** " + ", ".join(f"`{t}`" for t in tags))
                    else:
                        st.markdown("_No tags_")
                with mc3:
                    st.markdown(f"**Summary:** {summary}" if summary else "_No summary_")


def _render_upsert_stage(data: Dict[str, Any]) -> None:
    """Render Upsert stage: per-store details with chunk mapping."""
    dense_store = data.get("dense_store", {})
    sparse_store = data.get("sparse_store", {})
    image_store = data.get("image_store", {})

    # ── Overview metrics ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dense Vectors", dense_store.get("count", data.get("vector_count", 0)))
    with c2:
        st.metric("Sparse (BM25)", sparse_store.get("count", data.get("bm25_docs", 0)))
    with c3:
        st.metric("Images", image_store.get("count", data.get("images_indexed", 0)))

    # ── Dense store details ──
    if dense_store:
        with st.expander("🟦 Dense Vector Store (ChromaDB)", expanded=True):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**Backend:** `{dense_store.get('backend', '—')}`")
                st.markdown(f"**Collection:** `{dense_store.get('collection', '—')}`")
            with dc2:
                st.markdown(f"**Path:** `{dense_store.get('path', '—')}`")
                st.markdown(f"**Vectors:** {dense_store.get('count', 0)}")

    # ── Sparse store details ──
    if sparse_store:
        with st.expander("🟨 Sparse Index (BM25)", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Backend:** `{sparse_store.get('backend', '—')}`")
                st.markdown(f"**Collection:** `{sparse_store.get('collection', '—')}`")
            with sc2:
                st.markdown(f"**Path:** `{sparse_store.get('path', '—')}`")
                st.markdown(f"**Documents:** {sparse_store.get('count', 0)}")

    # ── Image store details ──
    if image_store and image_store.get("count", 0) > 0:
        with st.expander(f"🖼️ Image Storage ({image_store.get('count', 0)} images)", expanded=True):
            st.markdown(f"**Backend:** `{image_store.get('backend', '—')}`")
            imgs = image_store.get("images", [])
            if imgs:
                img_rows = [
                    {
                        "Image ID": img.get("image_id", ""),
                        "Page": img.get("page", 0),
                        "File": img.get("file_path", ""),
                        "Doc Hash": img.get("doc_hash", "")[:16] + "…",
                    }
                    for img in imgs
                ]
                st.table(img_rows)

    # ── Chunk → Vector ID mapping ──
    chunk_mapping = data.get("chunk_mapping", [])
    if chunk_mapping:
        with st.expander(f"🔗 Chunk → Vector Mapping ({len(chunk_mapping)} entries)", expanded=False):
            mapping_rows = [
                {
                    "#": i + 1,
                    "Chunk ID": m.get("chunk_id", ""),
                    "Vector ID": m.get("vector_id", ""),
                    "Store": m.get("store", ""),
                    "Collection": m.get("collection", ""),
                }
                for i, m in enumerate(chunk_mapping)
            ]
            st.table(mapping_rows)

    # ── Fallback: legacy format with just vector_ids ──
    if not chunk_mapping and not dense_store:
        vector_ids = data.get("vector_ids", [])
        if vector_ids:
            with st.expander("Vector IDs", expanded=False):
                for vid in vector_ids:
                    st.code(vid, language=None)
