#!/usr/bin/env python
"""Compare two evaluate.py report.json files (baseline vs experiment)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def full_score(cr: float, faith: float, cp: float) -> float:
    return 0.4 * cr + 0.3 * faith + 0.3 * cp


def main() -> int:
    baseline = PROJECT_ROOT / "results" / "baseline_v2" / "report.json"
    experiment = PROJECT_ROOT / "results" / "p1.1-full48" / "report.json"
    if len(sys.argv) >= 3:
        baseline = Path(sys.argv[1])
        experiment = Path(sys.argv[2])

    b = load_report(baseline)
    e = load_report(experiment)
    bm, em = b["aggregate_metrics"], e["aggregate_metrics"]

    print("=" * 60)
    print(f"Baseline:   {baseline}")
    print(f"Experiment: {experiment}")
    print("=" * 60)
    for key in ("context_recall", "context_precision", "faithfulness"):
        bv, ev = bm.get(key), em.get(key)
        if bv is None or ev is None:
            continue
        delta = ev - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<22} {bv:.4f} -> {ev:.4f}  ({sign}{delta:.4f})")

    bs = b.get("score_full") or full_score(bm["context_recall"], bm["faithfulness"], bm["context_precision"])
    es = e.get("score_full") or full_score(em["context_recall"], em["faithfulness"], em["context_precision"])
    print(f"  {'score_full':<22} {bs:.4f} -> {es:.4f}  ({'+' if es >= bs else ''}{es - bs:.4f})")

    # Boundary type faith
    def by_type_faith(report: dict, type_name: str) -> float | None:
        for qr in report.get("query_results", []):
            pass
        bt = report.get("by_type_metrics", {})
        if type_name in bt:
            return bt[type_name].get("faithfulness")
        return None

    for tname in ("边界无答案", "公式定理定位", "跨章节关联"):
        bv_t = by_type_faith(b, tname)
        ev_t = by_type_faith(e, tname)
        if bv_t is not None and ev_t is not None:
            print(f"  {tname} Faith:        {bv_t:.4f} -> {ev_t:.4f}  ({ev_t - bv_t:+.4f})")

    improved = es > bs + 1e-6
    print(f"\n  Verdict: {'✅ improved' if improved else '❌ not improved'} (score_full)")
    return 0 if improved else 1


if __name__ == "__main__":
    sys.exit(main())
