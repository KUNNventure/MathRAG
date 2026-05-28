"""DataService – read-only facade for browsing ingested data.

Wraps ``DocumentManager``, ``ChromaStore``, and ``ImageStorage`` to
provide the data the Data Browser page needs, without coupling the
UI to storage internals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Safety cap when scanning Chroma metadatas for the Data Browser.
_MAX_CHROMA_SCAN = 200_000


class DataService:
    """Provide read-only access to ingested documents, chunks, and images.

    Lazily instantiates the heavy storage objects on first call so that
    importing the module alone has zero cost.
    """

    def __init__(self) -> None:
        self._manager: Any = None
        self._chroma: Any = None
        self._images: Any = None
        self._current_collection: str = ""
        self._last_chroma_scan: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_stores(self, collection: Optional[str] = None) -> None:
        """Create storage objects on first use.

        Args:
            collection: Optional collection name. The ChromaStore will be
                        re-created if the requested collection differs from
                        the currently loaded one.
        """
        target_collection = collection or "default"

        # Re-create chroma if collection changed
        if (
            self._manager is not None
            and self._current_collection == target_collection
        ):
            return

        from src.core.settings import load_settings, resolve_path
        from src.ingestion.document_manager import DocumentManager
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import ImageStorage
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings()

        chroma = VectorStoreFactory.create(
            settings, collection_name=target_collection
        )
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{target_collection}")))
        images = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images")),
        )
        integrity = SQLiteIntegrityChecker(
            db_path=str(resolve_path("data/db/ingestion_history.db"))
        )

        self._chroma = chroma
        self._images = images
        self._manager = DocumentManager(chroma, bm25, images, integrity)
        self._current_collection = target_collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """Return all available ChromaDB collection names."""
        try:
            from src.core.settings import load_settings, resolve_path
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            settings = load_settings()
            persist_dir = str(
                resolve_path(settings.vector_store.persist_directory)
            )
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            return sorted(c.name for c in client.list_collections())
        except Exception as exc:
            logger.warning("Failed to list collections: %s", exc)
            return ["default"]

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return ingested documents as plain dicts (UI-friendly).

        Each dict has keys: source_path, source_hash, collection,
        chunk_count, image_count, processed_at.
        """
        self._ensure_stores(collection)
        from dataclasses import asdict

        docs = self._manager.list_documents(collection)
        return [asdict(d) for d in docs]

    def list_documents_from_chroma(
        self, collection: Optional[str] = None, *, max_chunks_to_scan: int = _MAX_CHROMA_SCAN
    ) -> List[Dict[str, Any]]:
        """Aggregate documents by scanning Chroma chunk metadatas (not SQLite).

        Use this when vectors exist but ``ingestion_history`` has no matching
        rows.  Each dict matches ``list_documents`` shape for UI reuse, plus
        optional ``grades`` and ``content_types`` (lists of str) for filtering.
        ``processed_at`` is always ``None`` (unknown from Chroma alone).
        """
        self._ensure_stores(collection)
        coll = self._chroma.collection
        coll_name = collection or self._current_collection or "default"
        try:
            total = int(coll.count())
        except Exception as exc:
            logger.warning("Chroma count failed: %s", exc)
            self._last_chroma_scan = {"total": 0, "scanned": 0, "complete": False, "capped": False}
            return []

        if total == 0:
            self._last_chroma_scan = {"total": 0, "scanned": 0, "complete": True}
            return []

        batch_size = 2000
        scanned = 0
        # doc_key -> aggregated row (mutated in loop)
        groups: Dict[str, Dict[str, Any]] = {}

        offset = 0
        while offset < total and scanned < max_chunks_to_scan:
            try:
                batch = coll.get(
                    limit=min(batch_size, total - offset, max_chunks_to_scan - scanned),
                    offset=offset,
                    include=["metadatas"],
                )
            except Exception as exc:
                logger.warning("Chroma get batch failed at offset %s: %s", offset, exc)
                break

            ids = batch.get("ids") or []
            metas = batch.get("metadatas") or []
            if not ids:
                break

            for meta in metas:
                if not isinstance(meta, dict):
                    meta = {}
                doc_hash = meta.get("doc_hash")
                source_path = str(meta.get("source_path", "") or "").strip()
                doc_key = str(doc_hash).strip() if doc_hash else source_path or "unknown"

                row = groups.get(doc_key)
                if row is None:
                    grades: Set[str] = set()
                    ctypes: Set[str] = set()
                    row = {
                        "source_path": source_path or "—",
                        "source_hash": str(doc_hash) if doc_hash else "",
                        "collection": coll_name,
                        "chunk_count": 0,
                        "image_count": 0,
                        "processed_at": None,
                        "grades": grades,
                        "content_types": ctypes,
                    }
                    groups[doc_key] = row
                row["chunk_count"] = int(row["chunk_count"]) + 1
                if source_path and row.get("source_path") in ("", "—"):
                    row["source_path"] = source_path
                if not row.get("source_hash") and doc_hash:
                    row["source_hash"] = str(doc_hash)

                g = row["grades"]
                ct = row["content_types"]
                assert isinstance(g, set) and isinstance(ct, set)
                gv = meta.get("grade")
                if gv is not None and str(gv).strip() != "":
                    g.add(str(gv).strip())
                ctv = meta.get("content_type")
                if ctv is not None and str(ctv).strip() != "":
                    ct.add(str(ctv).strip().lower())

            n = len(ids)
            offset += n
            scanned += n
            if n < batch_size:
                break

        out: List[Dict[str, Any]] = []
        for row in groups.values():
            sh = row.get("source_hash") or ""
            if isinstance(sh, str) and sh:
                try:
                    row["image_count"] = len(self._images.list_images(doc_hash=sh))
                except Exception as exc:
                    logger.debug("list_images failed for %s: %s", sh[:16], exc)
                    row["image_count"] = 0
            grades_set = row.pop("grades", set())
            ct_set = row.pop("content_types", set())
            row["grades"] = sorted(grades_set)
            row["content_types"] = sorted(ct_set)
            out.append(row)

        out.sort(key=lambda r: (Path(str(r.get("source_path", ""))).name.lower()))
        self._last_chroma_scan = {
            "total": total,
            "scanned": scanned,
            "complete": scanned >= total,
            "capped": scanned >= max_chunks_to_scan and total > scanned,
        }
        return out

    def chroma_chunk_count(self, collection: Optional[str] = None) -> int:
        """Return Chroma ``collection.count()`` for the active collection."""
        self._ensure_stores(collection)
        try:
            return int(self._chroma.collection.count())
        except Exception as exc:
            logger.warning("chroma_chunk_count failed: %s", exc)
            return 0

    def list_ingestion_history(
        self,
        collection: Optional[str] = None,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Recent success/failed rows from ``ingestion_history`` (SQLite)."""
        self._ensure_stores(collection)
        return self._manager.list_ingestion_history(collection, limit=limit)

    def get_document_detail(
        self, doc_id: str, collection: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return document detail as a plain dict, or None."""
        self._ensure_stores(collection)
        from dataclasses import asdict

        detail = self._manager.get_document_detail(doc_id)
        if detail is None:
            return None
        return asdict(detail)

    def get_chunks(
        self,
        source_hash: str,
        collection: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return chunk records from ChromaDB matching *source_hash* (``doc_hash``).

        If *source_hash* is empty, rows matched by ``source_path`` in Chroma metadata
        (legacy / edge cases).  Pass ``source_path=`` explicitly when needed.

        Each dict has keys: id, text, metadata.
        """
        self._ensure_stores(collection)
        try:
            where: Dict[str, Any]
            if source_hash:
                where = {"doc_hash": source_hash}
            elif source_path:
                where = {"source_path": str(source_path)}
            else:
                return []

            kwargs: Dict[str, Any] = {
                "where": where,
                "include": ["documents", "metadatas"],
            }
            if limit is not None:
                kwargs["limit"] = int(limit)

            results = self._chroma.collection.get(**kwargs)
            chunks: List[Dict[str, Any]] = []
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            for i, cid in enumerate(ids):
                chunks.append(
                    {
                        "id": cid,
                        "text": docs[i] if docs else "",
                        "metadata": metas[i] if metas else {},
                    }
                )
            return chunks
        except Exception as exc:
            logger.warning("Failed to get chunks for %s: %s", source_hash, exc)
            return []

    def get_images(
        self, source_hash: str, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return image records for a document."""
        self._ensure_stores(collection)
        try:
            return self._images.list_images(doc_hash=source_hash)
        except Exception as exc:
            logger.warning("Failed to get images for %s: %s", source_hash, exc)
            return []

    def delete_document(
        self,
        source_path: str,
        collection: Optional[str] = None,
        source_hash: Optional[str] = None,
    ) -> Any:
        """Delete a document via the underlying DocumentManager.

        Returns a ``DeleteResult`` dataclass.
        """
        self._ensure_stores(collection)
        return self._manager.delete_document(
            source_path,
            collection or "default",
            source_hash=source_hash,
        )

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return aggregate stats as a plain dict."""
        self._ensure_stores(collection)
        from dataclasses import asdict

        stats = self._manager.get_collection_stats(collection)
        return asdict(stats)

    def reset_all(self) -> Dict[str, Any]:
        """Delete ALL data: ChromaDB collections, BM25 indexes, images, integrity DB, and trace logs.

        Returns a summary dict with counts of what was deleted.
        """
        import shutil
        from src.core.settings import load_settings, resolve_path
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        summary: Dict[str, Any] = {
            "collections_deleted": 0,
            "bm25_cleared": False,
            "images_cleared": False,
            "integrity_cleared": False,
            "traces_cleared": False,
            "errors": [],
        }

        settings = load_settings()

        # 1. Delete all ChromaDB collections
        try:
            persist_dir = str(resolve_path(settings.vector_store.persist_directory))
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            colls = client.list_collections()
            for c in colls:
                client.delete_collection(c.name)
                summary["collections_deleted"] += 1
        except Exception as exc:
            summary["errors"].append(f"ChromaDB: {exc}")

        # 2. Clear BM25 indexes (remove entire bm25 directory)
        try:
            bm25_dir = resolve_path("data/db/bm25")
            if bm25_dir.exists():
                shutil.rmtree(bm25_dir)
                bm25_dir.mkdir(parents=True, exist_ok=True)
            summary["bm25_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"BM25: {exc}")

        # 3. Clear image storage (SQLite DB + image files)
        try:
            img_db = resolve_path("data/db/image_index.db")
            if img_db.exists():
                img_db.unlink()
            img_dir = resolve_path("data/images")
            if img_dir.exists():
                shutil.rmtree(img_dir)
                img_dir.mkdir(parents=True, exist_ok=True)
            summary["images_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"Images: {exc}")

        # 4. Clear file integrity database
        try:
            integrity_db = resolve_path("data/db/ingestion_history.db")
            if integrity_db.exists():
                integrity_db.unlink()
            summary["integrity_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"Integrity: {exc}")

        # 5. Clear trace logs
        try:
            traces_file = resolve_path("logs/traces.jsonl")
            if traces_file.exists():
                traces_file.unlink()
            summary["traces_cleared"] = True
        except Exception as exc:
            summary["errors"].append(f"Traces: {exc}")

        # Reset internal state so next call re-initializes
        self._manager = None
        self._chroma = None
        self._images = None
        self._current_collection = ""

        return summary
