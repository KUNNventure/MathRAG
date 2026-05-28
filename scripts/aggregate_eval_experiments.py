#!/usr/bin/env python
"""Aggregate all results/*/ experiments into _MASTER_SUMMARY.json and optional CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.observability.dashboard.services.eval_results_service import (  # noqa: E402
    build_master_summary,
    save_master_summary,
)


def write_csv(summary: dict, path: Path) -> None:
    rows = summary.get("experiments") or []
    if not rows:
        return
    fields = [
        "run_id", "query_count", "collection", "chunk_size", "chunk_count", "top_k",
        "context_recall", "context_precision", "faithfulness",
        "score_partial", "score_full", "elapsed_seconds",
        "faith_only", "cr_cp_only", "prompt_path", "answer_prompt", "judge_model",
        "gen_tokens_est", "judge_tokens_est", "total_tokens_est", "cost_est_cny",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            tok = row.get("token_estimate") or {}
            w.writerow({
                **row,
                "gen_tokens_est": tok.get("gen_tokens_est"),
                "judge_tokens_est": tok.get("judge_tokens_est"),
                "total_tokens_est": tok.get("total_tokens_est"),
                "cost_est_cny": tok.get("cost_est_cny"),
            })


def main() -> int:
    out = save_master_summary()
    data = json.loads(out.read_text(encoding="utf-8"))
    csv_path = out.parent / "_MASTER_SUMMARY.csv"
    write_csv(data, csv_path)
    print(f"Wrote {out} ({data['experiment_count']} experiments)")
    print(f"Wrote {csv_path}")

    desktop = Path(r"C:\Users\xsk\Desktop\RAG项目优化\实验汇总_MASTER.json")
    if desktop.parent.is_dir():
        desktop.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied to {desktop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
