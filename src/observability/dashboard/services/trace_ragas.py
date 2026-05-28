"""Helpers to run Ragas on a single stored query trace (dashboard)."""

from __future__ import annotations

from typing import Any, Dict, List


def pick_trace_chunks_for_eval(stages_by_name: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer rerank → fusion → dense → sparse chunk lists."""
    for stage in ("rerank", "fusion", "dense_retrieval", "sparse_retrieval"):
        block = stages_by_name.get(stage) or {}
        data = block.get("data") or {}
        chunks = data.get("chunks")
        if isinstance(chunks, list) and chunks:
            return chunks
    return []


def trace_chunks_to_evaluator_payload(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "chunk_id": str(c.get("chunk_id", "") or ""),
            }
        )
    return out


def fallback_answer_from_chunks(chunks: List[Dict[str, Any]], max_chars: int = 4500) -> str:
    """Cheap placeholder when no model answer exists (Ragas still needs non-empty answer)."""
    parts: List[str] = []
    for c in chunks:
        t = (c.get("text") or "").strip()
        if t:
            parts.append(t)
    if not parts:
        return ""
    joined = "\n\n---\n\n".join(parts)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n…（已截断；建议填写真实模型回答以获得有意义指标）"
    return joined


def run_ragas_for_trace(
    query: str,
    raw_chunks: List[Dict[str, Any]],
    generated_answer: str,
    *,
    judge_model: str | None = None,
) -> Dict[str, float]:
    """Run configured Ragas evaluator on trace-derived query/context/answer."""
    from dataclasses import replace

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory

    if not query.strip():
        raise ValueError("Query is empty.")
    payload = trace_chunks_to_evaluator_payload(raw_chunks)
    if not payload:
        raise ValueError("No chunk text in trace for evaluation.")
    if not (generated_answer or "").strip():
        raise ValueError("generated_answer is required for Ragas.")

    settings = load_settings()
    ev_cfg = settings.evaluation
    settings_ragas = replace(
        settings,
        evaluation=replace(ev_cfg, enabled=True, provider="ragas"),
    )

    kwargs: Dict[str, Any] = {}
    if judge_model and judge_model.strip():
        kwargs["judge_model"] = judge_model.strip()

    evaluator = EvaluatorFactory.create(settings_ragas, **kwargs)
    return evaluator.evaluate(
        query=query.strip(),
        retrieved_chunks=payload,
        generated_answer=generated_answer.strip(),
    )
