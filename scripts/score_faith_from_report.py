#!/usr/bin/env python
"""Score faithfulness only from a saved report (same chunks + answers).

Usage:
    python scripts/score_faith_from_report.py --source-dir results/p1.5-full48-v34
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env_bootstrap import ensure_repo_dotenv_loaded
from scripts._faith_adjust_lib import compute_faith_adjusted

ensure_repo_dotenv_loaded()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS faithfulness on saved report answers.")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--judge-model", default="qwen-turbo")
    p.add_argument("--collection", default=None)
    return p.parse_args()


def fetch_chunks(collection: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
    from src.core.settings import load_settings
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    store = VectorStoreFactory.create(load_settings(), collection_name=collection)
    records = store.get_by_ids(chunk_ids)
    return [{"text": r.get("text", ""), "metadata": r.get("metadata", {})} for r in records]


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    report_path = source / "report.json"
    if not report_path.is_file():
        print(f"Missing {report_path}", file=sys.stderr)
        return 1

    config_path = source / "config.json"
    config: Dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    collection = args.collection or config.get("collection") or "math_textbooks"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    query_rows = report.get("query_results") or []

    from src.core.settings import load_settings
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(
        settings=load_settings(),
        metrics=["faithfulness"],
        judge_model=args.judge_model,
    )

    t0 = time.monotonic()
    updated: List[Dict[str, Any]] = []
    faith_vals: List[float] = []

    for i, qr in enumerate(query_rows, 1):
        query = qr["query"]
        chunk_ids = qr.get("retrieved_chunk_ids") or []
        answer = qr.get("generated_answer") or "."
        chunks = fetch_chunks(collection, chunk_ids)
        print(f"[{i}/{len(query_rows)}] Faith: {query[:50]}...", flush=True)
        try:
            metrics = evaluator.evaluate(
                query=query,
                retrieved_chunks=chunks,
                generated_answer=answer,
                ground_truth=None,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            metrics = {}

        row = dict(qr)
        row["metrics"] = {**(row.get("metrics") or {}), **{k: round(v, 4) for k, v in metrics.items()}}
        updated.append(row)
        if "faithfulness" in row["metrics"]:
            faith_vals.append(row["metrics"]["faithfulness"])

    elapsed = time.monotonic() - t0
    faith_mean = round(sum(faith_vals) / len(faith_vals), 4) if faith_vals else 0.0
    adjust_rows = [
        {
            "query": r["query"],
            "answer": r.get("generated_answer", ""),
            "type": r.get("question_type", ""),
            "metrics": r.get("metrics", {}),
        }
        for r in updated
    ]
    faith_adj = compute_faith_adjusted(adjust_rows)

    report["evaluator_name_faith"] = "RagasEvaluator"
    report["faith_scored_from"] = str(source)
    report["faith_elapsed_seconds"] = round(elapsed, 1)
    report["aggregate_metrics"] = {"faithfulness": faith_mean}
    report["faithfulness_mean"] = faith_mean
    report["faith_adjusted"] = faith_adj
    report["query_results"] = updated
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    answers_path = source / "answers.jsonl"
    with answers_path.open("w", encoding="utf-8") as f:
        for r in updated:
            rec = {
                "query": r["query"],
                "type": r.get("question_type", ""),
                "difficulty": r.get("difficulty", ""),
                "answer": r.get("generated_answer", ""),
                "metrics": r.get("metrics", {}),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "faith_raw": faith_mean,
        "faith_adjusted": faith_adj,
        "faith_adjusted_boundary5_weida": faith_adj["faith_adjusted_boundary5_weida"],
    }
    (source / "faith_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Faith raw={faith_mean}")
    print(f"  Faith adj (boundary5_weida)={faith_adj['faith_adjusted_boundary5_weida']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
