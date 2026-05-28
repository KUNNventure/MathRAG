"""One-off: compare Phase 2 c1500 runs vs p1.1 baseline."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(report: dict) -> tuple[float, float, float, dict]:
    agg = report["aggregate_metrics"]
    cr, cp = agg["context_recall"], agg["context_precision"]
    sp = report.get("score_partial", 0.6 * cr + 0.4 * cp)
    by = report.get("by_question_type") or report.get("by_type_metrics") or {}
    return cr, cp, sp, by


def per_query(report: dict) -> dict[str, dict]:
    return {q["query"]: q["metrics"] for q in report["query_results"]}


def main() -> None:
    runs = [
        ("c1000 k=10 (p1.1)", ROOT / "results/p1.1/report.json"),
        ("c1500 k=5", ROOT / "results/p2-c1500-k5/report.json"),
        ("c1500 k=7", ROOT / "results/p2-c1500-k7/report.json"),
    ]
    for label, path in runs:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        r = load_report(path)
        cr, cp, sp, by = metrics(r)
        coll = r.get("params", {}).get("collection", "?")
        tk = r.get("params", {}).get("top_k", "?")
        print(f"\n=== {label} | {coll} top_k={tk} ===")
        print(f"CR={cr:.4f} CP={cp:.4f} Score_p={sp:.4f}")
        for t in sorted(by):
            v = by[t]
            print(f"  {t}: CR={v['context_recall']:.3f} CP={v['context_precision']:.3f}")

    p11 = load_report(ROOT / "results/p1.1/report.json")
    p25 = load_report(ROOT / "results/p2-c1500-k5/report.json")
    r1, r2 = per_query(p11), per_query(p25)
    print("\n=== Per-query delta (c1500 k5 - c1000 k10) ===")
    rows = []
    for q in sorted(set(r1) | set(r2)):
        m1, m2 = r1.get(q, {}), r2.get(q, {})
        dcr = m2.get("context_recall", 0) - m1.get("context_recall", 0)
        dcp = m2.get("context_precision", 0) - m1.get("context_precision", 0)
        if abs(dcr) > 0.05 or abs(dcp) > 0.05:
            rows.append((dcr, dcp, q))
    for dcr, dcp, q in sorted(rows):
        flag = " ⚠️" if dcr <= -0.2 or (dcr == 0 and dcp == 0 and "边界" not in q) else ""
        print(f"  dCR={dcr:+.2f} dCP={dcp:+.2f} | {q[:50]}{flag}")


if __name__ == "__main__":
    main()
