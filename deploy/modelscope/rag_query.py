"""RAG query API for ModelScope Gradio demo (hybrid search + rerank + answer)."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_PUBLISHER_RE = re.compile(r"【([^】]+)】")
_GRADE_RE = re.compile(r"(七|八|九)年级(?:(上|下)册)?")
_NOISE_SUFFIXES = ("数学电子课本", "电子课本", "数学课本", "课本", "数学")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"


def _ensure_api_keys() -> None:
    """Load secrets from ModelScope / .env (no keys in settings.yaml)."""
    from src.core.api_keys import load_dotenv_if_present, resolve_api_key

    load_dotenv_if_present()
    key = resolve_api_key(provider="qwen")
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)
        os.environ.setdefault("DASHSCOPE_API_KEY", key)


@lru_cache(maxsize=1)
def _get_settings():
    from src.core.settings import load_settings

    _ensure_api_keys()
    return load_settings(str(_SETTINGS_PATH))


@lru_cache(maxsize=1)
def _get_pipeline():
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.reranker import create_core_reranker
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.response.answer_prompt import build_answer_messages
    from src.core.settings import resolve_path
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.llm import LLMFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.observability.logger import configure_logging

    settings = _get_settings()
    configure_logging(settings.observability.log_level)
    collection = settings.vector_store.collection_name or "default"

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    bm25_indexer = BM25Indexer(
        index_dir=str(resolve_path(f"data/db/bm25/{collection}"))
    )
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection

    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    hybrid_search.fusion = RRFFusion(k=settings.retrieval.rrf_k)
    reranker = create_core_reranker(settings=settings)

    return {
        "settings": settings,
        "hybrid_search": hybrid_search,
        "reranker": reranker,
        "build_answer_messages": build_answer_messages,
        "llm_factory": LLMFactory,
    }


def _result_metadata(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        meta = result.get("metadata", {})
        return meta if isinstance(meta, dict) else {}
    meta = getattr(result, "metadata", {}) or {}
    return meta if isinstance(meta, dict) else {}


def _result_field(result: Any, field: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


def format_source_label(source_path: str) -> str:
    """Short citation label from path or filename.

    Example::

        【人教版】八年级下册数学电子课本.pdf → 人教版·八年级下册
    """
    raw = (source_path or "").strip()
    if not raw or raw in ("未知来源", "unknown"):
        return "未知来源"

    stem = Path(raw).stem
    publisher: str | None = None
    pub_match = _PUBLISHER_RE.search(stem)
    if pub_match:
        publisher = pub_match.group(1).strip()
        stem = _PUBLISHER_RE.sub("", stem).strip()

    grade_match = _GRADE_RE.search(stem)
    if grade_match:
        grade = f"{grade_match.group(1)}年级"
        if grade_match.group(2):
            grade += f"{grade_match.group(2)}册"
        if publisher:
            return f"{publisher}·{grade}"
        return grade

    for noise in _NOISE_SUFFIXES:
        stem = stem.replace(noise, "")
    stem = stem.strip("【】· _-\t")
    if publisher:
        return f"{publisher}·{stem}" if stem else publisher
    return stem or Path(raw).name


def _format_sources(results: List[Any], limit: int = 5) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for result in results[:limit]:
        meta = _result_metadata(result)
        source_path = meta.get("source_path") or meta.get("source") or ""
        source = format_source_label(str(source_path)) if source_path else "未知来源"
        content = (_result_field(result, "text", "") or "").strip()
        sources.append({"source": source, "content": content})
    return sources


def query(
    question: str,
    *,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    use_rerank: bool = True,
) -> Dict[str, Any]:
    """Run full RAG: hybrid retrieval, optional rerank, grounded answer generation."""
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入问题", "sources": []}

    pipe = _get_pipeline()
    settings = pipe["settings"]
    hybrid_search = pipe["hybrid_search"]
    reranker = pipe["reranker"]

    retrieve_top_k = settings.retrieval.fusion_top_k
    output_top_k = top_k if top_k is not None else settings.rerank.top_k

    hybrid_detail = hybrid_search.search(
        query=question,
        top_k=retrieve_top_k,
        filters=None,
        trace=None,
        return_details=True,
    )
    fused = list(hybrid_detail.results)

    if use_rerank and reranker.is_enabled and fused:
        rerank_result = reranker.rerank(
            query=question,
            results=fused,
            top_k=output_top_k,
            trace=None,
        )
        results = list(rerank_result.results)[:output_top_k]
    else:
        results = fused[:output_top_k]

    if not results:
        return {
            "answer": "未检索到相关内容，请换一种问法再试。",
            "sources": [],
        }

    context_blocks: List[str] = []
    for idx, result in enumerate(results, start=1):
        meta = _result_metadata(result)
        source = meta.get("source_path", "unknown")
        chunk_id = _result_field(result, "chunk_id", "")
        text = (_result_field(result, "text", "") or "").strip()
        if text:
            context_blocks.append(f"[{idx}] source={source} chunk_id={chunk_id}\n{text}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "（无可用上下文）"
    messages = pipe["build_answer_messages"](question, context_text, settings=settings)
    llm = pipe["llm_factory"].create(settings)
    answer = llm.chat(messages).content.strip()

    return {
        "answer": answer,
        "sources": _format_sources(results),
    }
