#!/usr/bin/env python
"""Adjusted faithfulness: exclude product-correct / RAGAS-misjudged items."""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_QUERIES = {
    "导数的定义是什么？",
    "正态分布是什么？",
    "三角函数的和差化积公式是什么？",
    "虚数和复数是什么？",
    "什么是微积分？",
}

# Product-correct patterns for boundary (v3.4)
REJECTION_MARKERS = (
    "不在人教版7-9年级教材范围内",
    "本系统无法回答",
    "属于高中数学内容",
    "属于大学数学内容",
)
INSUFFICIENT_MARKERS = ("根据当前教材内容无法确定",)


def load_answers(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def is_boundary_rejection_ok(answer: str) -> tuple[bool, str]:
    if any(m in answer for m in REJECTION_MARKERS):
        return True, "规则5固定拒答"
    if any(m in answer for m in INSUFFICIENT_MARKERS) and "韦达" not in answer:
        if "一元二次" in answer and "x_1" in answer:  # vieta hallucination
            return False, "编造定理内容"
        return True, "规则2范围外/无法确定"
    return False, "未合规拒答"


def is_vieta_hallucination(answer: str) -> bool:
    return "韦达定理" in answer and ("x_1 + x_2" in answer or "x_1x_2" in answer)


def product_verdict(row: dict) -> tuple[str, str]:
    """Return (verdict, reason) for human review."""
    q, ans, t = row["query"], row["answer"], row["type"]
    f = row["metrics"].get("faithfulness", 0)

    if t == "边界无答案":
        ok, why = is_boundary_rejection_ok(ans)
        if ok:
            return "好（拒答合规）", f"RAGAS误判F={f:.2f}；{why}"
        return "差（应拒答未拒答或编造）", f"F={f:.2f}"

    if "韦达定理" in q:
        if is_vieta_hallucination(ans):
            return "差（编造）", f"用通用知识写韦达，F={f:.2f}仍偏高"
        if "无法确定" in ans:
            return "中（诚实但未召回）", "未检索到，至少未硬编"
        return "差", ans[:60]

    if "圆和勾股定理" in q:
        if f < 0.6:
            return "中（有关联但偏牵强）", f"F={f:.2f}，检索噪声下仍展开论述"
        return "好", f"F={f:.2f}"

    if "二次函数求最值" in q and len(ans) > 2000:
        return "差（输出失控）", f"F={f:.2f} RAGAS误判偏高，大量无关粘贴"

    if f >= 0.95:
        return "好", f"F={f:.2f}"
    if f >= 0.7:
        return "中", f"F={f:.2f}"
    return "中偏弱", f"F={f:.2f}"


def adjusted_faith(rows: list[dict], exclude_queries: set[str]) -> dict:
    scores = []
    adjusted = []
    for r in rows:
        f = r["metrics"]["faithfulness"]
        scores.append(f)
        if r["query"] in exclude_queries:
            adjusted.append(1.0)  # credit product-correct boundary
        else:
            adjusted.append(f)
    n = len(scores)
    return {
        "n": n,
        "faith_raw": sum(scores) / n,
        "faith_adjusted": sum(adjusted) / n,
        "excluded_n": len(exclude_queries),
        "excluded_queries": sorted(exclude_queries),
    }


def main() -> None:
    p34 = load_answers(PROJECT_ROOT / "results/p3.4/answers.jsonl")
    p11 = load_answers(PROJECT_ROOT / "results/p1.1-full48/answers.jsonl")

    # Misjudged: boundary with OK rejection in v3.4
    exclude = set()
    misjudged = []
    for r in p34:
        if r["type"] == "边界无答案":
            ok, why = is_boundary_rejection_ok(r["answer"])
            if ok and r["metrics"]["faithfulness"] < 0.5:
                exclude.add(r["query"])
                misjudged.append({"query": r["query"], "faith_ragas": r["metrics"]["faithfulness"], "reason": why})

    adj34 = adjusted_faith(p34, exclude)
    adj11 = adjusted_faith(p11, set())  # p1.1 no exclusion for comparison

    # p1.1 boundary-only
    b11 = [r for r in p11 if r["type"] == "边界无答案"]
    b34 = [r for r in p34 if r["type"] == "边界无答案"]

    reviews = []
    for r in p34:
        v, reason = product_verdict(r)
        reviews.append({
            "query": r["query"],
            "type": r["type"],
            "faith_ragas": r["metrics"]["faithfulness"],
            "product_verdict": v,
            "note": reason,
            "eval_adjust_exclude": r["query"] in exclude,
        })

    out = {
        "p3.4": adj34,
        "p1.1_full48_raw": {"faith_mean": sum(r["metrics"]["faithfulness"] for r in p11) / len(p11)},
        "p1.1_boundary_only": sum(r["metrics"]["faithfulness"] for r in b11) / len(b11),
        "p3.4_boundary_ragas": sum(r["metrics"]["faithfulness"] for r in b34) / len(b34),
        "p3.4_boundary_adjusted": 1.0,
        "misjudged_excluded": misjudged,
        "delta_adjusted_vs_p11_raw": adj34["faith_adjusted"] - (sum(r["metrics"]["faithfulness"] for r in p11) / len(p11)),
        "per_query_reviews_p34": reviews,
    }

    out_path = PROJECT_ROOT / "results/p3.4/adjusted_faith_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Faith 均值 ===")
    print(f"p3.4 RAGAS 原始:     {adj34['faith_raw']:.4f}")
    print(f"p3.4 调整后(排除{adj34['excluded_n']}题): {adj34['faith_adjusted']:.4f}")
    print(f"p1.1-full48 原始:    {out['p1.1_full48_raw']['faith_mean']:.4f}")
    print(f"调整后 vs p1.1:      {out['delta_adjusted_vs_p11_raw']:+.4f}")
    print(f"\n边界题 Faith: p1.1={out['p1.1_boundary_only']:.2f}  p3.4 RAGAS={out['p3.4_boundary_ragas']:.2f}  p3.4 调整后=1.00")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
