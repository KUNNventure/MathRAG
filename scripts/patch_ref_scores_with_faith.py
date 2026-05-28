#!/usr/bin/env python
"""Attach Faith to *-ref-cr-cp reports and compute score_full.

Rules (user-approved):
- E001b: CR/CP@ref + Faith from baseline_v2 (same generated answers in source report).
- E004b: CR/CP@ref + Faith **adjusted** from p3.4-r2 (faith_adjusted_boundary5_weida).
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMULA = "0.4*CR + 0.3*Faith + 0.3*CP"

PATCHES = [
    {
        "ref_dir": PROJECT_ROOT / "results/baseline_v2-ref-cr-cp",
        "source_dir": PROJECT_ROOT / "results/baseline_v2",
        "label": "E001b",
        "faith_mode": "raw",
        "rerank": "v0 英文",
        "prompt": "v1",
        "faith_note": "同次 baseline 答案；仅 CR/CP 改为 @reference",
    },
    {
        "ref_dir": PROJECT_ROOT / "results/p3.4-r2-ref-cr-cp",
        "source_dir": PROJECT_ROOT / "results/p3.4-r2",
        "label": "E004b",
        "faith_mode": "adjusted",
        "rerank": "v2 中文",
        "prompt": "v3.4",
        "faith_note": "faith_adjusted_boundary5_weida（边界拒答+韦达合规计1）",
    },
]


def score_full(cr: float, cp: float, faith: float) -> float:
    return round(0.4 * cr + 0.3 * faith + 0.3 * cp, 4)


def load_faith(source: Path, mode: str) -> tuple[float, str, float | None]:
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    raw = float(report["aggregate_metrics"]["faithfulness"])
    if mode == "raw":
        return raw, "faithfulness (RAGAS raw)", raw
    adj_block = report.get("faith_adjusted") or {}
    adjusted = float(adj_block.get("faith_adjusted_boundary5_weida", raw))
    return adjusted, adj_block.get("method", "faith_adjusted_boundary5_weida"), raw


def patch_ref(entry: dict) -> dict:
    ref_dir: Path = entry["ref_dir"]
    source_dir: Path = entry["source_dir"]
    report_path = ref_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    agg = report["aggregate_metrics"]
    cr, cp = float(agg["context_recall"]), float(agg["context_precision"])
    faith, faith_kind, faith_raw = load_faith(source_dir, entry["faith_mode"])
    sf = score_full(cr, cp, faith)
    report["scoring_note_faith"] = entry["faith_note"]
    report["faithfulness"] = faith
    report["faithfulness_raw"] = faith_raw
    report["faith_source"] = str(source_dir)
    report["faith_kind"] = faith_kind
    report["score_full"] = sf
    report["score_full_formula"] = FORMULA
    agg["faithfulness"] = faith
    report["aggregate_metrics"] = agg
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = ref_dir / "rescore_meta.json"
    meta_obj = {}
    if meta_path.is_file():
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_obj.update(
        {
            "faithfulness": faith,
            "faithfulness_raw": faith_raw,
            "faith_kind": faith_kind,
            "score_full": sf,
            "score_full_formula": FORMULA,
        }
    )
    meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": entry["label"],
        "cr": cr,
        "cp": cp,
        "faith": faith,
        "faith_raw": faith_raw,
        "faith_kind": faith_kind,
        "score_partial": report.get("score_partial"),
        "score_full": sf,
        "rerank": entry["rerank"],
        "prompt": entry["prompt"],
        "dir": str(ref_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "valid_combined_score": True,
        "note": entry["faith_note"],
    }


def main() -> int:
    combined = [patch_ref(p) for p in PATCHES]

    legacy = []
    for label, path, note in [
        ("E001", "baseline_v2", "旧 CR/CP 口径，同次全量"),
        ("E002", "p1.1-full48", "旧 CR/CP 口径，同次全量"),
        ("E005", "final-v3.4", "旧 CR/CP 口径，同次全量"),
    ]:
        r = json.loads((PROJECT_ROOT / "results" / path / "report.json").read_text(encoding="utf-8"))
        a = r["aggregate_metrics"]
        legacy.append(
            {
                "id": label,
                "cr": a.get("context_recall"),
                "cp": a.get("context_precision"),
                "faith": a.get("faithfulness"),
                "score_full": r.get("score_full"),
                "dir": f"results/{path}/",
                "metric_note": note,
            }
        )

    e004_raw_faith = 0.8189
    e004b = combined[1]
    score_raw_faith = score_full(e004b["cr"], e004b["cp"], e004_raw_faith)

    summary = {
        "formula": FORMULA,
        "combined_at_ref_valid": combined,
        "e004b_faith_raw_score": score_raw_faith,
        "legacy_same_run": legacy,
        "deltas": {
            "E004b_minus_E001b_score": round(combined[1]["score_full"] - combined[0]["score_full"], 4),
            "E004b_minus_E001b_score_p": round(combined[1]["score_partial"] - combined[0]["score_partial"], 4),
            "E001b_minus_E001_legacy_score": round(combined[0]["score_full"] - legacy[0]["score_full"], 4),
        },
        "invalid_do_not_use": [
            "E004b CR/CP + E005 Faith（final 另次全量 Faith，非调整口径）",
            "任意 @ref CR/CP + 与源答案不一致来源的 Faith",
        ],
    }

    out = PROJECT_ROOT / "results" / "_SCORES_AT_REF.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    desktop = Path(r"C:\Users\xsk\Desktop\RAG项目优化\SCORES_AT_REF.json")
    if desktop.parent.is_dir():
        desktop.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {out}")
    for row in combined:
        print(
            f"  {row['id']:6} CR={row['cr']} CP={row['cp']} "
            f"F={row['faith']} (raw={row['faith_raw']}) Score={row['score_full']}"
        )
    print(f"  E004b w/ raw F=0.8189 → Score={score_raw_faith} (对比用)")
    print(f"  Δ Score E004b-E001b = {summary['deltas']['E004b_minus_E001b_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
