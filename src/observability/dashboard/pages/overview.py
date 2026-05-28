"""Overview page – collection stats, chunk quality, and system config at a glance."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


@st.cache_resource(show_spinner=False)
def _get_chroma_client():
    """Cached ChromaDB client – avoids re-creating every render."""
    from src.core.settings import load_settings, resolve_path
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings = load_settings()
    persist_dir = str(resolve_path(settings.vector_store.persist_directory))
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


@st.cache_data(ttl=30, show_spinner=False)
def _get_collection_stats():
    """Per-collection stats: chunks, LLM ratio, content types, grades."""
    client = _get_chroma_client()
    stats = {}
    for col_info in client.list_collections():
        name = col_info.name if hasattr(col_info, "name") else str(col_info)
        coll = client.get_collection(name)
        total = coll.count()
        if total == 0:
            stats[name] = {"total": 0, "llm_refine": 0, "rule_refine": 0,
                           "llm_enrich": 0, "rule_enrich": 0,
                           "content_types": {}, "grades": {}}
            continue

        # Sample up to 2000 chunks for metadata stats
        results = coll.get(limit=min(total, 2000), include=["metadatas"])
        metas = results.get("metadatas", [])

        llm_refine = sum(1 for m in metas if m.get("refined_by") == "llm")
        rule_refine = sum(1 for m in metas if m.get("refined_by") == "rule")
        llm_enrich = sum(1 for m in metas if m.get("enriched_by") == "llm")
        rule_enrich = sum(1 for m in metas if m.get("enriched_by") == "rule")

        # Content type distribution
        content_types = {}
        for m in metas:
            ct = m.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1

        # Grade distribution
        grades = {}
        for m in metas:
            g = m.get("grade", "?")
            grades[g] = grades.get(g, 0) + 1

        stats[name] = {
            "total": total,
            "llm_refine": llm_refine, "rule_refine": rule_refine,
            "llm_enrich": llm_enrich, "rule_enrich": rule_enrich,
            "content_types": content_types, "grades": grades,
        }
    return stats


@st.cache_data(ttl=10, show_spinner=False)
def _get_trace_counts():
    """Lightweight trace stats – just counts by type."""
    from src.core.settings import resolve_path
    from pathlib import Path
    tp = resolve_path("logs/traces.jsonl")
    if not tp.exists():
        return {"ingestion": 0, "query": 0, "total": 0}
    import json
    ingestion = query = 0
    with tp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if t.get("trace_type") == "ingestion":
                    ingestion += 1
                else:
                    query += 1
            except json.JSONDecodeError:
                pass
    return {"ingestion": ingestion, "query": query, "total": ingestion + query}


def render() -> None:
    st.header("System Overview")

    # ── Row 1: Quick metrics ───────────────────────────────────────
    stats = _get_collection_stats()
    trace_counts = _get_trace_counts()

    col1, col2, col3, col4 = st.columns(4)
    total_chunks = sum(s["total"] for s in stats.values())
    total_llm_refine = sum(s["llm_refine"] for s in stats.values())
    total_rule_refine = sum(s["rule_refine"] for s in stats.values())
    refine_pct = f"{total_llm_refine / max(total_llm_refine + total_rule_refine, 1) * 100:.0f}%"

    with col1:
        st.metric("Collections", len(stats))
    with col2:
        st.metric("Total Chunks", total_chunks)
    with col3:
        st.metric("ChunkRefiner LLM", refine_pct,
                  help=f"LLM: {total_llm_refine} / Rule: {total_rule_refine}")
    with col4:
        st.metric("Traces (I/Q)",
                  f"{trace_counts['ingestion']}/{trace_counts['query']}",
                  help="ingestion / query traces")

    st.divider()

    # ── Row 2: Collection breakdown ────────────────────────────────
    st.subheader("Collection Breakdown")
    if not stats:
        st.info("No data yet. Go to Ingestion Manager to import PDFs.")
    else:
        for name, s in sorted(stats.items()):
            if s["total"] == 0:
                st.caption(f"**{name}** — empty")
                continue

            with st.expander(
                f"**{name}** — {s['total']} chunks  "
                f"| Refine: {s['llm_refine']}L/{s['rule_refine']}R  "
                f"| Enrich: {s['llm_enrich']}L/{s['rule_enrich']}R",
                expanded=(len(stats) == 1)
            ):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.caption("ChunkRefiner")
                    refine_total = s["llm_refine"] + s["rule_refine"]
                    if refine_total > 0:
                        llm_r = s["llm_refine"] / refine_total
                        st.progress(llm_r, text=f"LLM {llm_r:.0%}")
                        if s["rule_refine"] > 0:
                            st.caption(f"{s['rule_refine']} fallback")

                    st.caption("MetadataEnricher")
                    enrich_total = s["llm_enrich"] + s["rule_enrich"]
                    if enrich_total > 0:
                        llm_e = s["llm_enrich"] / enrich_total
                        st.progress(llm_e, text=f"LLM {llm_e:.0%}")

                with c2:
                    st.caption("Content Types")
                    for ct, count in sorted(s["content_types"].items(),
                                            key=lambda x: -x[1]):
                        st.text(f"{ct}: {count}")

                with c3:
                    st.caption("Grades")
                    for g, count in sorted(s["grades"].items()):
                        st.text(f"Grade {g}: {count}")

    st.divider()

    # ── Row 3: Component config ────────────────────────────────────
    st.subheader("Component Configuration")
    try:
        cards = ConfigService().get_component_cards()
    except Exception as exc:
        st.error(f"Failed to load config: {exc}")
        return

    # Show only active components
    active = [c for c in cards if c.name not in ("Vision LLM",)]
    cols = st.columns(min(len(active), 3))
    for idx, card in enumerate(active):
        with cols[idx % 3]:
            st.markdown(f"**{card.name}**")
            st.caption(f"{card.provider} · `{card.model}`")
