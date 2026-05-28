"""Scan ``results/*/`` experiment dirs for reports, configs, answers, and derived stats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Heuristic $/1M tokens (CNY-ish order of magnitude for budgeting docs only).
_PRICE_GEN_PER_1M = 2.0
_PRICE_JUDGE_PER_1M = 0.3
_PRICE_RERANK_PER_1M = 2.0

_COLLECTION_CHUNK: Dict[str, tuple[int, int]] = {
    "math_textbooks": (1000, 721),
    "math_textbooks_c500": (500, 1795),
    "math_textbooks_c1500": (1500, 543),
}

_CHARS_PER_TOKEN_ZH = 1.8


@dataclass
class ExperimentSummary:
    run_id: str
    run_dir: str
    report_path: Optional[str] = None
    config_path: Optional[str] = None
    answers_path: Optional[str] = None
    has_report: bool = False
    query_count: int = 0
    test_set_path: str = ""
    collection: str = ""
    chunk_size: Optional[int] = None
    chunk_count: Optional[int] = None
    top_k: Optional[int] = None
    dense_top_k: Optional[int] = None
    sparse_top_k: Optional[int] = None
    rrf_k: Optional[int] = None
    rerank_enabled: Optional[bool] = None
    judge_model: str = ""
    prompt_path: Optional[str] = None
    answer_prompt: Optional[str] = None
    cr_cp_only: bool = False
    faith_only: bool = False
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    faithfulness: Optional[float] = None
    score_partial: Optional[float] = None
    score_full: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    total_elapsed_ms: Optional[float] = None
    token_estimate: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def results_root(root: Optional[Path] = None) -> Path:
    from src.core.settings import resolve_path

    return root if root is not None else resolve_path("results")


def list_experiment_dirs(root: Optional[Path] = None) -> List[Path]:
    base = results_root(root)
    if not base.is_dir():
        return []
    dirs: List[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        if (p / "report.json").is_file() or (p / "config.json").is_file():
            dirs.append(p)
    return dirs


def _infer_chunk(collection: str) -> tuple[Optional[int], Optional[int]]:
    if collection in _COLLECTION_CHUNK:
        cs, cnt = _COLLECTION_CHUNK[collection]
        return cs, cnt
    m = re.search(r"_c(\d+)$", collection)
    if m:
        return int(m.group(1)), None
    return None, None


def _estimate_tokens(
    report: Dict[str, Any],
    answers_path: Optional[Path],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    n = int(report.get("query_count") or len(report.get("query_results") or []))
    if n == 0 and answers_path and answers_path.is_file():
        n = sum(1 for _ in answers_path.open(encoding="utf-8"))

    top_k = int(params.get("top_k") or 10)
    collection = str(params.get("collection") or "math_textbooks")
    chunk_size, _ = _infer_chunk(collection)
    chunk_size = chunk_size or 1000

    gen_chars = 0
    answer_lines = 0
    if answers_path and answers_path.is_file():
        for line in answers_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            answer_lines += 1
            try:
                rec = json.loads(line)
                gen_chars += len(rec.get("answer") or "")
            except json.JSONDecodeError:
                continue

    if gen_chars == 0:
        for qr in report.get("query_results") or []:
            gen_chars += len(qr.get("generated_answer") or "")

    ctx_chars = 0
    for qr in report.get("query_results") or []:
        n_chunks = len(qr.get("retrieved_chunk_ids") or [])
        ctx_chars += n_chunks * min(chunk_size, 1200)

    faith_only = bool(params.get("faith_only"))
    cr_cp_only = bool(params.get("cr_cp_only"))
    if faith_only:
        ragas_calls = n * 1
    elif cr_cp_only:
        ragas_calls = n * 2
    else:
        ragas_calls = n * 3

    rerank_on = params.get("rerank_enabled", True)
    rerank_chars = n * top_k * 400 if rerank_on else 0

    gen_tokens = int(gen_chars / _CHARS_PER_TOKEN_ZH)
    ctx_tokens = int(ctx_chars / _CHARS_PER_TOKEN_ZH)
    judge_tokens = int(ragas_calls * 2500)
    rerank_tokens = int(rerank_chars / _CHARS_PER_TOKEN_ZH)

    cost_gen = gen_tokens / 1_000_000 * _PRICE_GEN_PER_1M
    cost_judge = judge_tokens / 1_000_000 * _PRICE_JUDGE_PER_1M
    cost_rerank = rerank_tokens / 1_000_000 * _PRICE_RERANK_PER_1M

    return {
        "query_count": n,
        "gen_chars": gen_chars,
        "context_chars_est": ctx_chars,
        "gen_tokens_est": gen_tokens,
        "context_tokens_est": ctx_tokens,
        "judge_tokens_est": judge_tokens,
        "rerank_tokens_est": rerank_tokens,
        "total_tokens_est": gen_tokens + judge_tokens + rerank_tokens,
        "cost_est_cny": round(cost_gen + cost_judge + cost_rerank, 2),
        "ragas_metric_calls": ragas_calls,
        "note": "估算值：中文按1.8字/token；判分按题×指标×~2500 token；未含 embedding",
    }


def summarize_experiment(run_dir: Path) -> ExperimentSummary:
    run_id = run_dir.name
    summary = ExperimentSummary(run_id=run_id, run_dir=str(run_dir).replace("\\", "/"))

    report_path = run_dir / "report.json"
    config_path = run_dir / "config.json"
    answers_path = run_dir / "answers.jsonl"

    if report_path.is_file():
        summary.report_path = str(report_path)
        summary.has_report = True
    if config_path.is_file():
        summary.config_path = str(config_path)
    if answers_path.is_file():
        summary.answers_path = str(answers_path)

    params: Dict[str, Any] = {}
    if config_path.is_file():
        try:
            params = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary.errors.append(f"config.json: {exc}")

    report: Dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary.errors.append(f"report.json: {exc}")
            return summary

    if not params and report.get("params"):
        params = report["params"]

    agg = report.get("aggregate_metrics") or {}
    summary.query_count = int(report.get("query_count") or len(report.get("query_results") or []))
    summary.test_set_path = str(report.get("test_set_path") or params.get("test_set") or "")
    summary.collection = str(params.get("collection") or "")
    cs, cnt = _infer_chunk(summary.collection)
    summary.chunk_size = cs
    summary.chunk_count = cnt
    summary.top_k = params.get("top_k")
    summary.dense_top_k = params.get("dense_top_k")
    summary.sparse_top_k = params.get("sparse_top_k")
    summary.rrf_k = params.get("rrf_k")
    summary.rerank_enabled = params.get("rerank_enabled")
    summary.judge_model = str(params.get("judge_model") or "")
    summary.prompt_path = params.get("prompt_path")
    summary.answer_prompt = report.get("answer_prompt")
    summary.cr_cp_only = bool(params.get("cr_cp_only"))
    summary.faith_only = bool(params.get("faith_only"))
    summary.context_recall = agg.get("context_recall")
    summary.context_precision = agg.get("context_precision")
    summary.faithfulness = agg.get("faithfulness")
    summary.score_partial = report.get("score_partial")
    summary.score_full = report.get("score_full")
    summary.elapsed_seconds = report.get("elapsed_seconds")
    summary.total_elapsed_ms = report.get("total_elapsed_ms")
    if summary.elapsed_seconds is None and summary.total_elapsed_ms:
        summary.elapsed_seconds = round(summary.total_elapsed_ms / 1000, 1)

    summary.token_estimate = _estimate_tokens(report, answers_path if answers_path.is_file() else None, params)
    return summary


def build_master_summary(root: Optional[Path] = None) -> Dict[str, Any]:
    dirs = list_experiment_dirs(root)
    experiments = [summarize_experiment(d).to_dict() for d in dirs]
    experiments.sort(key=lambda e: e.get("run_id") or "")
    return {
        "generated_by": "eval_results_service.build_master_summary",
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def load_master_summary(root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = results_root(root) / "_MASTER_SUMMARY.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_master_summary(root: Optional[Path] = None) -> Path:
    data = build_master_summary(root)
    path = results_root(root) / "_MASTER_SUMMARY.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
