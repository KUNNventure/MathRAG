#!/usr/bin/env python
"""Print baseline vs p1.1-full48 comparison for doc sync."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def full_score(cr: float, f: float, cp: float) -> float:
    return 0.4 * cr + 0.3 * f + 0.3 * cp


def main() -> None:
    b = json.loads((PROJECT_ROOT / "results/baseline_v2/report.json").read_text(encoding="utf-8"))
    e = json.loads((PROJECT_ROOT / "results/p1.1-full48/report.json").read_text(encoding="utf-8"))
    bm, em = b["aggregate_metrics"], e["aggregate_metrics"]
    bs = b.get("score_full") or full_score(bm["context_recall"], bm["faithfulness"], bm["context_precision"])
    es = e["score_full"]
    print("AGGREGATE")
    for k, label in (
        ("context_recall", "CR"),
        ("context_precision", "CP"),
        ("faithfulness", "Faith"),
    ):
        print(f"  {label}: {bm[k]:.4f} -> {em[k]:.4f} ({em[k] - bm[k]:+.4f})")
    print(f"  Score: {bs:.4f} -> {es:.4f} ({es - bs:+.4f})")
    print(f"  Score_p: {0.6 * bm['context_recall'] + 0.4 * bm['context_precision']:.4f} -> {e['score_partial']:.4f}")
    print("\nBY_TYPE")
    order = [
        "知识点定义",
        "例题检索",
        "知识点覆盖范围",
        "跨章节关联",
        "易混概念辨析",
        "章节定位",
        "公式定理定位",
        "边界无答案",
    ]
    for t in order:
        bt, et = b["by_type_metrics"][t], e["by_type_metrics"][t]
        bs_t = full_score(bt["context_recall"], bt["faithfulness"], bt["context_precision"])
        es_t = full_score(et["context_recall"], et["faithfulness"], et["context_precision"])
        print(
            f"  {t}: Score {bs_t:.4f}->{es_t:.4f} ({es_t - bs_t:+.4f}) | "
            f"CR {bt['context_recall']:.3f}->{et['context_recall']:.3f} | "
            f"CP {bt['context_precision']:.3f}->{et['context_precision']:.3f} | "
            f"Faith {bt['faithfulness']:.3f}->{et['faithfulness']:.3f}"
        )


if __name__ == "__main__":
    main()
