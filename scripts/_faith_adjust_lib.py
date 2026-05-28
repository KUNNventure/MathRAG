"""Faith adjustment rules (boundary5 + 韦达) shared by scoring pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

BOUNDARY_QUERIES = {
    "导数的定义是什么？",
    "正态分布是什么？",
    "三角函数的和差化积公式是什么？",
    "虚数和复数是什么？",
    "什么是微积分？",
}

REJECTION_MARKERS = (
    "不在人教版7-9年级教材范围内",
    "本系统无法回答",
    "属于高中数学内容",
    "属于大学数学内容",
)
INSUFFICIENT_MARKERS = ("根据当前教材内容无法确定",)


def is_boundary_rejection_ok(answer: str) -> Tuple[bool, str]:
    if any(m in answer for m in REJECTION_MARKERS):
        return True, "规则5固定拒答"
    if any(m in answer for m in INSUFFICIENT_MARKERS) and "韦达" not in answer:
        return True, "规则2范围外/无法确定"
    return False, "未合规拒答"


def is_vieta_honest(answer: str) -> bool:
    return "韦达" in answer and any(m in answer for m in INSUFFICIENT_MARKERS)


def is_vieta_hallucination(answer: str) -> bool:
    return "韦达定理" in answer and ("x_1 + x_2" in answer or "x_1x_2" in answer)


def adjusted_score_for_query(query: str, answer: str, faith_raw: float, qtype: str = "") -> float:
    """Per-query adjusted faith (E004b rules)."""
    if qtype == "边界无答案" or query in BOUNDARY_QUERIES:
        ok, _ = is_boundary_rejection_ok(answer)
        if ok and faith_raw < 0.5:
            return 1.0
    if "韦达定理" in query and is_vieta_honest(answer) and not is_vieta_hallucination(answer):
        return 1.0
    return faith_raw


def compute_faith_adjusted(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate faith_raw / boundary5 / boundary5_weida from query rows."""
    raw_scores: List[float] = []
    b5_scores: List[float] = []
    b5w_scores: List[float] = []
    non_boundary: List[float] = []

    for row in rows:
        faith_raw = float((row.get("metrics") or {}).get("faithfulness", 0))
        query = row.get("query") or ""
        answer = row.get("generated_answer") or row.get("answer") or ""
        qtype = row.get("question_type") or row.get("type") or ""

        raw_scores.append(faith_raw)

        adj_b5 = faith_raw
        if qtype == "边界无答案" or query in BOUNDARY_QUERIES:
            ok, _ = is_boundary_rejection_ok(answer)
            if ok and faith_raw < 0.5:
                adj_b5 = 1.0
        b5_scores.append(adj_b5)

        adj_weida = adj_b5
        if "韦达定理" in query and is_vieta_honest(answer) and not is_vieta_hallucination(answer):
            adj_weida = 1.0
        b5w_scores.append(adj_weida)

        if qtype != "边界无答案" and query not in BOUNDARY_QUERIES:
            non_boundary.append(faith_raw)

    n = len(raw_scores) or 1
    return {
        "method": "边界5题合规拒答F=0计1；韦达仅无法确定计1",
        "faith_raw": round(sum(raw_scores) / n, 4),
        "faith_adjusted_boundary5": round(sum(b5_scores) / n, 4),
        "faith_adjusted_boundary5_weida": round(sum(b5w_scores) / n, 4),
        "faith_non_boundary_only": round(sum(non_boundary) / len(non_boundary), 4)
        if non_boundary
        else None,
    }
