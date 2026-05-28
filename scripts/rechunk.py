#!/usr/bin/env python
"""Re-chunk documents from an existing collection at a different chunk_size.

Reads chunk texts from a source collection, reconstructs per-document full text,
then re-chunks and re-runs the LLM pipeline (ChunkRefiner + MetadataEnricher +
Embedding + Storage) into a target collection.

Usage:
    python scripts/rechunk.py --source math_textbooks --target math_textbooks_c500
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Re-chunk documents from existing collection.")
    p.add_argument("--source", required=True, help="Source collection name.")
    p.add_argument("--target", required=True, help="Target collection name.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from src.core.settings import load_settings
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.ingestion.chunking.document_chunker import DocumentChunker
    from src.ingestion.transform.chunk_refiner import ChunkRefiner
    from src.ingestion.transform.metadata_enricher import MetadataEnricher
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.ingestion.embedding.dense_encoder import DenseEncoder
    from src.ingestion.embedding.sparse_encoder import SparseEncoder
    from src.ingestion.embedding.batch_processor import BatchProcessor
    from src.ingestion.storage.vector_upserter import VectorUpserter
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.core.trace import TraceContext
    from src.core.types import Document, Chunk

    settings = load_settings()

    # ── Step 1: Read chunks from source ──────────────────────────────
    print(f"[*] Reading chunks from '{args.source}'...")
    source_vs = VectorStoreFactory.create(settings, collection_name=args.source)
    data = source_vs.collection.get(include=["documents", "metadatas"])

    if not data["ids"]:
        print("[FAIL] Source collection is empty.")
        return 2

    total = len(data["ids"])
    docs_map: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    for i in range(total):
        meta = data["metadatas"][i] or {}
        src = meta.get("source_path", "unknown")
        ci = meta.get("chunk_index", i)
        docs_map[src].append((ci, data["documents"][i], meta))

    print(f"   {total} chunks across {len(docs_map)} documents")

    # ── Step 2: Reconstruct full text per document ─────────────────
    documents: list[Document] = []
    for src, chunks in sorted(docs_map.items()):
        chunks.sort(key=lambda x: x[0])
        full_text = "\n\n".join(t for _, t, _ in chunks)
        meta = chunks[0][2]  # inherit first chunk's metadata
        documents.append(Document(
            id=f"rechunk_{abs(hash(src)) % (10**16):016x}",
            text=full_text,
            metadata={**meta, "source_path": src, "rechunked_from": args.source},
        ))
        if args.verbose:
            print(f"   {Path(src).name}: {len(chunks)} chunks → {len(full_text)} chars")

    # ── Step 3: Re-chunk ───────────────────────────────────────────
    chunker = DocumentChunker(settings)
    cs = settings.ingestion.chunk_size
    print(f"\n[*] Re-chunking at chunk_size={cs}...")
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))
    print(f"   {len(all_chunks)} chunks")

    # ── Step 4: ChunkRefiner (LLM) ──────────────────────────────────
    refiner = ChunkRefiner(settings)
    trace = TraceContext(trace_type="rechunk")
    if refiner.use_llm:
        print(f"\n[*] ChunkRefiner (LLM) on {len(all_chunks)} chunks...")
        all_chunks = refiner.transform(all_chunks, trace)
        refined = sum(1 for c in all_chunks if c.metadata.get("refined"))
        print(f"   {refined}/{len(all_chunks)} refined")
    else:
        print("\n[*] ChunkRefiner skipped (use_llm=False)")

    # ── Step 5: MetadataEnricher (LLM) ──────────────────────────────
    enricher = MetadataEnricher(settings)
    if enricher.use_llm:
        print(f"\n[*] MetadataEnricher (LLM) on {len(all_chunks)} chunks...")
        all_chunks = enricher.transform(all_chunks, trace)
        enriched = sum(1 for c in all_chunks if c.metadata.get("enriched"))
        print(f"   {enriched}/{len(all_chunks)} enriched")
    else:
        print("\n[*] MetadataEnricher skipped (use_llm=False)")

    # ── Step 6: Embed ──────────────────────────────────────────────
    print(f"\n[*] Encoding {len(all_chunks)} chunks...")
    embedding = EmbeddingFactory.create(settings)
    batch_size = settings.ingestion.batch_size if settings.ingestion else 100
    dense = DenseEncoder(embedding, batch_size=batch_size)
    sparse = SparseEncoder()
    processor = BatchProcessor(dense_encoder=dense, sparse_encoder=sparse, batch_size=batch_size)

    batch_results = processor.process(all_chunks)
    total_ok = sum(1 for br in batch_results if br.success)
    print(f"   {total_ok}/{len(batch_results)} batches ok")

    # ── Step 7: Store ──────────────────────────────────────────────
    print(f"\n[*] Storing to '{args.target}'...")
    upserter = VectorUpserter(settings, collection_name=args.target)
    bm25 = BM25Indexer(index_dir=f"data/db/bm25/{args.target}")

    stored = 0
    for br in batch_results:
        if br.success:
            upserter.upsert(br.embedded_chunks)
            bm25.index(br.embedded_chunks)
            stored += len(br.embedded_chunks)
    print(f"   {stored} chunks stored")

    print(f"\n[OK] Done. Collection '{args.target}': {stored} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
