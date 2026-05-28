#!/usr/bin/env python
"""RAG evaluation script — runs golden test set through HybridSearch + LLM generation + RAGAS scoring.

Usage:
    python scripts/evaluate.py                                    # default test set, default params
    python scripts/evaluate.py --test-set path/to/golden.json     # custom test set
    python scripts/evaluate.py --top-k 5 --no-rerank              # override params
    python scripts/evaluate.py --dense-only --top-k 10            # dense retrieval only
    python scripts/evaluate.py --dense-weight 0.3 --sparse-weight 0.7  # RRF list weights
    python scripts/evaluate.py --cr-cp-only                       # skip faithfulness (Phase 1)
    python scripts/evaluate.py --faith-only                       # faithfulness only (Phase 3)
    python scripts/evaluate.py --no-metrics                       # retrieve+generate only; then @ref pipeline
    python scripts/evaluate.py --workers 1                        # serial test cases (default is 4)
    python scripts/evaluate.py --results-dir results/baseline    # custom output dir

Results are saved to results/<timestamp>/, containing:
    - report.json       full metrics + per-type breakdown + per-query details
    - config.json       parameters used for this run
    - answers.jsonl     generated answers for each query (append-only across runs)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env_bootstrap import ensure_repo_dotenv_loaded

ensure_repo_dotenv_loaded()

# Phase 1 intermediate ranking (21-q subset).
PHASE1_SCORE_CR_WEIGHT = 0.6
PHASE1_SCORE_CP_WEIGHT = 0.4
# Full baseline / Phase 3 ranking (with Faith).
FULL_SCORE_CR_WEIGHT = 0.4
FULL_SCORE_FAITH_WEIGHT = 0.3
FULL_SCORE_CP_WEIGHT = 0.3


def phase1_score_p(cr: float, cp: float) -> float:
    return PHASE1_SCORE_CR_WEIGHT * cr + PHASE1_SCORE_CP_WEIGHT * cp


def full_score(cr: float, faith: float, cp: float) -> float:
    return (
        FULL_SCORE_CR_WEIGHT * cr
        + FULL_SCORE_FAITH_WEIGHT * faith
        + FULL_SCORE_CP_WEIGHT * cp
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAG evaluation against a golden test set.")
    p.add_argument("--test-set", default=r"C:\Users\xsk\Desktop\RAG项目优化\golden_test_set.json",
                   help="Path to golden test set JSON.")
    p.add_argument("--collection", default=None, help="Collection name.")
    p.add_argument("--top-k", type=int, default=10, help="fusion_top_k (default: 10).")
    p.add_argument("--dense-top-k", type=int, default=20)
    p.add_argument("--sparse-top-k", type=int, default=20)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--no-rerank", action="store_true", help="Disable reranker.")
    p.add_argument(
        "--dense-only",
        action="store_true",
        help="Disable BM25 / sparse retrieval (dense semantic search only).",
    )
    p.add_argument(
        "--dense-weight",
        type=float,
        default=None,
        help="RRF weight for dense ranking list (default: settings retrieval.dense_rrf_weight).",
    )
    p.add_argument(
        "--sparse-weight",
        type=float,
        default=None,
        help="RRF weight for sparse/BM25 ranking list (default: settings retrieval.sparse_rrf_weight).",
    )
    p.add_argument(
        "--cr-cp-only",
        action="store_true",
        help="Run only context_recall + context_precision (skip faithfulness; Phase 1).",
    )
    p.add_argument(
        "--faith-only",
        action="store_true",
        help="Run only faithfulness (skip CR/CP; Phase 3 prompt experiments).",
    )
    p.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip all RAGAS metrics (retrieve + generate only). Use with run_ref_eval_pipeline.py.",
    )
    p.add_argument("--no-generate", action="store_true",
                   help="Skip LLM answer generation (use chunk concatenation).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel test-case workers (default: 4). Use 1 for fully serial runs; "
                        "RAGAS judge + LLM rerank are serialised for thread safety.")
    p.add_argument("--judge-model", default="qwen-turbo",
                   help="LLM model for RAGAS judging (default: qwen-turbo, unified across all experiments).")
    p.add_argument("--results-dir", default=None,
                   help="Output directory (default: results/<YYYYMMDD_HHMMSS>/).")
    p.add_argument(
        "--prompt-path",
        default=None,
        help="Override answer generation prompt file (e.g. config/prompts/answer_generation_v3.4.txt).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metric_flags = sum([args.cr_cp_only, args.faith_only, args.no_metrics])
    if metric_flags > 1:
        print("❌ Use only one of --cr-cp-only, --faith-only, or --no-metrics.", file=sys.stderr)
        return 2

    # ── Configure logging (must happen before any logger use) ──────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # ── Load settings ──────────────────────────────────────────────
    try:
        from src.core.settings import load_settings
        print("[DEBUG] Loading settings...", flush=True)
        settings = load_settings()
        print("[DEBUG] Settings loaded OK", flush=True)
    except Exception as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 2

    # ── Override retrieval config ──────────────────────────────────
    if hasattr(settings, 'retrieval'):
        r = settings.retrieval
        dw = args.dense_weight if args.dense_weight is not None else r.dense_rrf_weight
        sw = args.sparse_weight if args.sparse_weight is not None else r.sparse_rrf_weight
        settings = replace(
            settings,
            retrieval=replace(
                r,
                fusion_top_k=args.top_k,
                dense_top_k=args.dense_top_k,
                sparse_top_k=args.sparse_top_k,
                rrf_k=args.rrf_k,
                dense_rrf_weight=dw,
                sparse_rrf_weight=sw,
            ),
        )
    if args.no_rerank and hasattr(settings, 'rerank'):
        settings = replace(settings, rerank=replace(settings.rerank, enabled=False))

    # ── Results folder ─────────────────────────────────────────────
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Results dir: {results_dir}", flush=True)

    # ── Build search stack ─────────────────────────────────────────
    print("[DEBUG] Building search stack...", flush=True)
    try:
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.hybrid_search import HybridSearchConfig, create_hybrid_search
        from src.libs.reranker.reranker_factory import RerankerFactory

        collection = args.collection or "math_textbooks"

        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        embedding_client = EmbeddingFactory.create(settings)

        dense_retriever = create_dense_retriever(
            settings=settings, embedding_client=embedding_client, vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=str(PROJECT_ROOT / "data" / "db" / "bm25" / collection))
        sparse_retriever = create_sparse_retriever(
            settings=settings, bm25_indexer=bm25_indexer, vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        ret = settings.retrieval
        hybrid_cfg = HybridSearchConfig(
            dense_top_k=ret.dense_top_k,
            sparse_top_k=ret.sparse_top_k,
            fusion_top_k=ret.fusion_top_k,
            enable_dense=True,
            enable_sparse=not args.dense_only,
            parallel_retrieval=True,
            metadata_filter_post=True,
            dense_rrf_weight=ret.dense_rrf_weight,
            sparse_rrf_weight=ret.sparse_rrf_weight,
        )
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            config=hybrid_cfg,
        )

        reranker = None
        if not args.no_rerank:
            try:
                reranker = RerankerFactory.create(settings)
                print(f"   Reranker: enabled ({type(reranker).__name__})")
            except Exception:
                print("   Reranker: unavailable, skipping")

        print(f"[DEBUG] Search stack ready (collection={collection}, top_k={args.top_k})", flush=True)
    except Exception as exc:
        print(f"❌ Failed to build search stack: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 2

    # ── Build evaluator ────────────────────────────────────────────
    evaluator = None
    if args.no_metrics:
        print("✅ Evaluator: skipped (--no-metrics); use run_ref_eval_pipeline.py for @ref CR/CP + Faith")
    else:
        try:
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator

            if args.cr_cp_only:
                eval_metrics = ["context_precision", "context_recall"]
            elif args.faith_only:
                eval_metrics = ["faithfulness"]
            else:
                eval_metrics = ["faithfulness", "context_precision", "context_recall"]
            evaluator = RagasEvaluator(
                settings=settings,
                metrics=eval_metrics,
                judge_model=args.judge_model,
            )
            print(f"✅ Evaluator: RagasEvaluator ({len(eval_metrics)} metrics, judge={args.judge_model})")
        except Exception as exc:
            print(f"❌ Evaluator error: {exc}", file=sys.stderr)
            return 2

    # ── Build answer generator ─────────────────────────────────────
    answer_generator = None
    if not args.no_generate:
        try:
            from src.core.response.answer_prompt import build_answer_messages
            from src.libs.llm.llm_factory import LLMFactory

            llm = LLMFactory.create(settings)

            def generate_answer(query: str, chunks: list) -> str:
                if not chunks:
                    return "（未检索到相关内容）"
                # Build context from chunks
                context_parts = []
                for i, c in enumerate(chunks):
                    text = c.text if hasattr(c, 'text') else str(c)
                    context_parts.append(f"[{i+1}] {text[:800]}")
                context = "\n\n".join(context_parts)

                messages = build_answer_messages(
                    query, context, settings=settings, prompt_path=args.prompt_path
                )
                try:
                    resp = llm.chat(messages)
                    text = resp.content if hasattr(resp, 'content') else str(resp)
                    return text.strip()
                except Exception:
                    # Fallback: concatenate chunk texts
                    return " ".join(c.text if hasattr(c, 'text') else str(c) for c in chunks[:3])

            answer_generator = generate_answer
            print(f"✅ Answer generator: {settings.llm.model}")
        except Exception as exc:
            print(f"⚠️  Answer generator unavailable ({exc}), using chunk concatenation")

    # ── Run evaluation ─────────────────────────────────────────────
    from src.observability.evaluation.eval_runner import EvalRunner
    print("[DEBUG] Creating EvalRunner...", flush=True)

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_generator=answer_generator,
        reranker=reranker,
        workers=args.workers,
    )

    worker_info = f", workers={args.workers}" if args.workers > 1 else ""
    print(f"\n[DEBUG] Running evaluation ({args.top_k=}, {args.dense_top_k=}, {args.sparse_top_k=}, {args.rrf_k=}{worker_info})...", flush=True)
    t_start = time.monotonic()

    try:
        print("[DEBUG] Starting runner.run()...", flush=True)
        report = runner.run(
            test_set_path=args.test_set,
            top_k=args.top_k,
            collection=collection,
        )
        print("[DEBUG] runner.run() completed", flush=True)
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.monotonic() - t_start

    # ── Print summary ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  EVALUATION REPORT")
    print("=" * 72)
    print(f"  Queries:    {len(report.query_results)}")
    print(f"  Time:       {elapsed:.0f}s")
    print(f"  Evaluator:  {report.evaluator_name}")
    print()

    if args.no_metrics:
        print("─" * 72)
        print("  NO RAGAS METRICS (retrieve + generate only)")
        print("─" * 72)
        print("  Next: python scripts/run_ref_eval_pipeline.py --source-dir", results_dir)
        print()

    # Aggregate
    if report.aggregate_metrics:
        print("─" * 72)
        print("  AGGREGATE METRICS")
        print("─" * 72)
    for metric, value in sorted(report.aggregate_metrics.items()):
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {metric:<25s} {bar} {value:.4f}")

    cr_agg = report.aggregate_metrics.get("context_recall")
    cp_agg = report.aggregate_metrics.get("context_precision")
    faith_agg = report.aggregate_metrics.get("faithfulness")
    if cr_agg is not None and cp_agg is not None:
        sp = phase1_score_p(cr_agg, cp_agg)
        print(f"  {'score_p (0.6CR+0.4CP)':<25s} {'█' * int(sp * 20)}{'░' * (20 - int(sp * 20))} {sp:.4f}")
    if cr_agg is not None and cp_agg is not None and faith_agg is not None:
        sf = full_score(cr_agg, faith_agg, cp_agg)
        print(f"  {'score (0.4CR+0.3F+0.3CP)':<25s} {'█' * int(sf * 20)}{'░' * (20 - int(sf * 20))} {sf:.4f}")

    # By type
    if report.by_type_metrics:
        print()
        print("─" * 72)
        print("  BY QUESTION TYPE")
        print("─" * 72)
        types = sorted(report.by_type_metrics.items())
        for tname, tmetrics in types:
            count = sum(1 for qr in report.query_results if qr.question_type == tname)
            parts = []
            for m, v in sorted(tmetrics.items()):
                parts.append(f"{m}={v:.3f}")
            print(f"  {tname:<20s} (n={count})  {' | '.join(parts)}")

    # Per query (abbreviated)
    print()
    if args.faith_only:
        sort_key = "faithfulness"
        print("─" * 72)
        print("  PER-QUERY (worst-first by faithfulness)")
        print("─" * 72)
        sorted_results = sorted(
            report.query_results,
            key=lambda qr: qr.metrics.get("faithfulness", 0),
        )
        for i, qr in enumerate(sorted_results, 1):
            f = qr.metrics.get("faithfulness", 0)
            flag = " ⚠️" if f < 0.5 else ""
            print(f"  [{i:2d}] F={f:.3f}  {qr.question_type}  {qr.query[:55]}{flag}")
    else:
        print("─" * 72)
        print("  PER-QUERY (worst-first by context_recall)")
        print("─" * 72)
        sorted_results = sorted(
            report.query_results,
            key=lambda qr: qr.metrics.get("context_recall", 0),
        )
        for i, qr in enumerate(sorted_results, 1):
            cr = qr.metrics.get("context_recall", 0)
            cp = qr.metrics.get("context_precision", 0)
            flag = " ⚠️" if cr < 0.3 else ""
            if "faithfulness" in qr.metrics:
                f = qr.metrics.get("faithfulness", 0)
                print(f"  [{i:2d}] CR={cr:.3f} CP={cp:.3f} F={f:.3f}  {qr.question_type}  {qr.query[:50]}{flag}")
            else:
                print(f"  [{i:2d}] CR={cr:.3f} CP={cp:.3f}      {qr.question_type}  {qr.query[:50]}{flag}")

    # ── Save results ───────────────────────────────────────────────
    report_dict = report.to_dict()
    report_dict["by_type_metrics"] = {
        k: {mk: round(mv, 4) for mk, mv in v.items()}
        for k, v in report.by_type_metrics.items()
    }
    report_dict["elapsed_seconds"] = round(elapsed, 1)
    if cr_agg is not None and cp_agg is not None:
        report_dict["score_partial"] = round(phase1_score_p(cr_agg, cp_agg), 4)
        report_dict["score_partial_formula"] = "0.6*CR + 0.4*CP"
    if cr_agg is not None and cp_agg is not None and faith_agg is not None:
        report_dict["score_full"] = round(full_score(cr_agg, faith_agg, cp_agg), 4)
        report_dict["score_full_formula"] = "0.4*CR + 0.3*Faith + 0.3*CP"
    report_dict["params"] = {
        "top_k": args.top_k,
        "dense_top_k": args.dense_top_k,
        "sparse_top_k": args.sparse_top_k,
        "rrf_k": args.rrf_k,
        "dense_rrf_weight": settings.retrieval.dense_rrf_weight,
        "sparse_rrf_weight": settings.retrieval.sparse_rrf_weight,
        "dense_only": args.dense_only,
        "rerank_enabled": not args.no_rerank,
        "generate_enabled": not args.no_generate,
        "judge_model": args.judge_model,
        "workers": args.workers,
        "collection": collection,
        "test_set": args.test_set,
        "cr_cp_only": args.cr_cp_only,
        "faith_only": args.faith_only,
        "no_metrics": args.no_metrics,
        "prompt_path": args.prompt_path,
    }
    if args.no_metrics:
        report_dict["scoring_note"] = (
            "retrieve+generate only; CR/CP@ref + Faith via run_ref_eval_pipeline.py"
        )
    if faith_agg is not None and args.faith_only:
        report_dict["faithfulness_mean"] = round(faith_agg, 4)
        report_dict["experiment"] = "phase3.4"

    with open(results_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(report_dict["params"], f, indent=2, ensure_ascii=False)

    # Append answers to answers.jsonl
    answers_path = results_dir / "answers.jsonl"
    with open(answers_path, "w", encoding="utf-8") as f:
        for qr in report.query_results:
            rec = {
                "query": qr.query,
                "type": qr.question_type,
                "difficulty": qr.difficulty,
                "answer": qr.generated_answer,
                "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n📁 Saved: {results_dir}/")
    print(f"   report.json    — full metrics")
    print(f"   config.json    — run parameters")
    print(f"   answers.jsonl  — generated answers")

    return 0


if __name__ == "__main__":
    sys.exit(main())
