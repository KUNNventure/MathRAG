#!/usr/bin/env python
"""Re-score CR/CP from a saved report (chunk IDs) + golden reference_answer — no retrieval/generation.

Usage:
    python scripts/rescore_ragas_cr_cp.py --source-dir results/p3.4-r2
    python scripts/rescore_ragas_cr_cp.py --source-dir results/p3.4-r2 --out-dir results/p3.4-r2-ref-cr-cp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env_bootstrap import ensure_repo_dotenv_loaded

ensure_repo_dotenv_loaded()

from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.evaluation.eval_runner import EvalRunner, GoldenTestCase
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rescore context_recall + context_precision@reference.")
    p.add_argument("--source-dir", required=True, help="Existing run with report.json (e.g. results/p3.4-r2)")
    p.add_argument("--out-dir", default=None, help="Output dir (default: <source>-ref-cr-cp)")
    p.add_argument("--test-set", default=None, help="Golden JSON (default: from source config.json)")
    p.add_argument("--judge-model", default="qwen-turbo")
    p.add_argument("--collection", default=None, help="Chroma collection (default: from config.json)")
    return p.parse_args()


def load_golden_by_query(path: Path) -> Dict[str, GoldenTestCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [GoldenTestCase.from_dict(row) for row in data.get("test_cases", [])]
    return {c.query: c for c in cases}


def fetch_chunks(collection: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    settings = load_settings()
    store = VectorStoreFactory.create(settings, collection_name=collection)
    records = store.get_by_ids(chunk_ids)
    return [{"text": r.get("text", ""), "metadata": r.get("metadata", {})} for r in records]


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    if not source.is_dir():
        print(f"Missing source dir: {source}", file=sys.stderr)
        return 1
    report_path = source / "report.json"
    if not report_path.is_file():
        print(f"Missing {report_path}", file=sys.stderr)
        return 1

    config_path = source / "config.json"
    config: Dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) if args.out_dir else Path(str(source) + "-ref-cr-cp")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_set = Path(args.test_set or config.get("test_set") or "")
    if not test_set.is_file():
        print("Golden test set path required (--test-set or config.json test_set)", file=sys.stderr)
        return 1

    collection = args.collection or config.get("collection") or "math_textbooks"
    golden = load_golden_by_query(test_set)
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    query_rows = old_report.get("query_results") or []

    settings = load_settings()
    evaluator = RagasEvaluator(
        settings=settings,
        metrics=["context_recall", "context_precision"],
        judge_model=args.judge_model,
    )

    t0 = time.monotonic()
    updated: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, float]] = []

    for i, qr in enumerate(query_rows, 1):
        query = qr["query"]
        tc = golden.get(query)
        if not tc or not tc.reference_answer:
            print(f"[{i}/{len(query_rows)}] skip (no reference_answer): {query[:40]}")
            continue

        chunk_ids = qr.get("retrieved_chunk_ids") or []
        answer = qr.get("generated_answer") or "."
        chunks = fetch_chunks(collection, chunk_ids)
        ground_truth = tc.ground_truth_payload()

        print(f"[{i}/{len(query_rows)}] {query[:50]}...", flush=True)
        try:
            metrics = evaluator.evaluate(
                query=query,
                retrieved_chunks=chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            metrics = {}

        row = dict(qr)
        row["metrics"] = {k: round(v, 4) for k, v in metrics.items()}
        row["metrics_legacy_faith"] = qr.get("metrics", {})
        updated.append(row)
        metric_rows.append(metrics)

    elapsed = time.monotonic() - t0
    agg = {k: round(v, 4) for k, v in aggregate_metrics(metric_rows).items()}
    sp = None
    cr, cp = agg.get("context_recall"), agg.get("context_precision")
    if cr is not None and cp is not None:
        sp = round(0.6 * cr + 0.4 * cp, 4)

    new_report = {
        "evaluator_name": "RagasEvaluator",
        "rescore_from": str(source),
        "scoring_note": "CR+CP@reference_answer; CP=ContextPrecision (not WithoutReference); no retrieval",
        "test_set_path": str(test_set),
        "collection": collection,
        "query_count": len(updated),
        "elapsed_seconds": round(elapsed, 1),
        "aggregate_metrics": agg,
        "score_partial": sp,
        "score_partial_formula": "0.6*CR + 0.4*CP",
        "query_results": updated,
    }

    out_report = out_dir / "report.json"
    out_report.write_text(json.dumps(new_report, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "source": str(source),
        "prior_faith_only": old_report.get("aggregate_metrics"),
        "new_cr_cp": agg,
    }
    (out_dir / "rescore_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone in {elapsed:.0f}s -> {out_dir}")
    print(f"  CR={agg.get('context_recall')} CP={agg.get('context_precision')} Score_p={sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
