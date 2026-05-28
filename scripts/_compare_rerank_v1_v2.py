"""Compare v2 (p1.1) vs p1.5 and baseline subset by type."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(r"C:\Users\xsk\Desktop\RAG项目优化\golden_subset_21.json")

# v1 per-query not archived; session aggregates:
V1_AGG = {"context_precision": 0.5007, "context_recall": 0.6901}
V2_AGG = {"context_precision": 0.4260, "context_recall": 0.8101}


def load_report(path: Path) -> dict[str, dict]:
    r = json.loads(path.read_text(encoding="utf-8"))
    return {qr["query"]: qr["metrics"] for qr in r["query_results"]}


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = golden["test_cases"] if isinstance(golden, dict) else golden
    meta_by_q = {c["query"]: c.get("meta", {}) for c in cases}

    v2 = load_report(ROOT / "results/p1.1/report.json")
    p15 = load_report(ROOT / "results/p1.5/report.json")
    subset = json.loads(
        (ROOT / "results/baseline_v2/subset_21_ragas.json").read_text(encoding="utf-8")
    )
    v0 = {x["query"]: x for x in subset["queries"]}

    rows = []
    for q, m2 in v2.items():
        meta = meta_by_q.get(q, {})
        m5 = p15.get(q, {})
        m0 = v0.get(q, {})
        rows.append(
            {
                "id": meta.get("id"),
                "type": meta.get("type", ""),
                "query": q,
                "cp_v2": m2.get("context_precision", 0),
                "cp_p15": m5.get("context_precision", 0),
                "cp_v0": m0.get("context_precision", 0),
                "cr_v2": m2.get("context_recall", 0),
                "cr_p15": m5.get("context_recall", 0),
            }
        )

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    print("=== v2 CP gap vs p1.5 (no rerank; ~same as v1+rerank on mean) ===\n")
    print(f"{'类型':14} n   CP_v2  CP_p15   ΔCP    CR_v2  CR_p15")
    type_deltas = []
    for t in sorted(by_type):
        g = by_type[t]
        n = len(g)
        cp2 = sum(x["cp_v2"] for x in g) / n
        cp5 = sum(x["cp_p15"] for x in g) / n
        cr2 = sum(x["cr_v2"] for x in g) / n
        cr5 = sum(x["cr_p15"] for x in g) / n
        dcp = cp2 - cp5
        type_deltas.append((dcp, t, n, cp2, cp5, cr2, cr5))
        print(f"{t:14} {n}  {cp2:.3f}  {cp5:.3f}  {dcp:+.3f}  {cr2:.2f}  {cr5:.2f}")

    print("\n=== v2 vs baseline v0 (English rerank) by type ===\n")
    for t in sorted(by_type):
        g = by_type[t]
        cp2 = sum(x["cp_v2"] for x in g) / len(g)
        cp0 = sum(x["cp_v0"] for x in g) / len(g)
        print(f"  {t:14} CP v2={cp2:.3f} v0={cp0:.3f} Δ={cp2-cp0:+.3f}")

    print("\n=== Worst v2 vs p15 per question (v2 CP lower) ===\n")
    for r in sorted(rows, key=lambda x: x["cp_v2"] - x["cp_p15"])[:10]:
        d = r["cp_v2"] - r["cp_p15"]
        qshort = r["query"][:36]
        print(
            f"  ID{r['id']:>3} {r['type'][:10]:10} "
            f"CP {r['cp_v2']:.3f} vs {r['cp_p15']:.3f} ({d:+.3f}) "
            f"CR {r['cr_v2']:.2f} vs {r['cr_p15']:.2f} | {qshort}"
        )

    print("\n=== Cross-chapter (v2 detail) ===\n")
    for r in rows:
        if r["type"] == "跨章节关联":
            print(f"  ID{r['id']:>3} CP={r['cp_v2']:.3f} CR={r['cr_v2']:.2f} | {r['query'][:40]}")

    print(f"\nAggregate v1 (session): CP={V1_AGG['context_precision']:.4f}")
    print(f"Aggregate v2 (p1.1):  CP={V2_AGG['context_precision']:.4f}")
    print(f"ΔCP total: {V2_AGG['context_precision']-V1_AGG['context_precision']:+.4f}")


if __name__ == "__main__":
    main()
