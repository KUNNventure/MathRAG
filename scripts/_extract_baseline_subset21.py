"""Extract baseline_v2 per-query RAGAS for golden_subset_21."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
subset = json.loads(
    Path(r"C:\Users\xsk\Desktop\RAG项目优化\golden_subset_21.json").read_text(encoding="utf-8")
)
baseline = json.loads((ROOT / "results/baseline_v2/report.json").read_text(encoding="utf-8"))

queries = {tc["query"]: tc["meta"] for tc in subset["test_cases"]}
by_query = {qr["query"]: qr for qr in baseline["query_results"]}

rows = []
missing = []
for q, meta in queries.items():
    qr = by_query.get(q)
    if not qr:
        missing.append({"id": meta["id"], "query": q})
        continue
    m = qr["metrics"]
    cr = m.get("context_recall", 0.0)
    cp = m.get("context_precision", 0.0)
    faith = m.get("faithfulness", 0.0)
    rows.append(
        {
            "id": meta["id"],
            "type": meta["type"],
            "difficulty": meta["difficulty"],
            "query": q,
            "context_recall": round(cr, 4),
            "context_precision": round(cp, 4),
            "faithfulness": round(faith, 4),
            "score_partial": round(0.6 * cr + 0.4 * cp, 4),
            "score_full": round(0.4 * cr + 0.3 * faith + 0.3 * cp, 4),
        }
    )

rows.sort(key=lambda r: r["id"])
out = {
    "source": "results/baseline_v2/report.json",
    "rerank_note": "baseline_v2 dir = experiment batch only; rerank prompt = v0 English 0-3 (rerank_en_v0.txt), NOT zh v2",
    "matched": len(rows),
    "missing": missing,
    "subset_aggregate": {},
    "by_type": {},
    "queries": rows,
}
if rows:
    n = len(rows)
    out["subset_aggregate"] = {
        "context_recall": round(sum(r["context_recall"] for r in rows) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4),
        "score_partial": round(sum(r["score_partial"] for r in rows) / n, 4),
        "score_full": round(sum(r["score_full"] for r in rows) / n, 4),
    }
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    for t, grp in sorted(by_type.items()):
        g = len(grp)
        out["by_type"][t] = {
            "n": g,
            "context_recall": round(sum(x["context_recall"] for x in grp) / g, 4),
            "context_precision": round(sum(x["context_precision"] for x in grp) / g, 4),
            "faithfulness": round(sum(x["faithfulness"] for x in grp) / g, 4),
        }

out_path = ROOT / "results" / "baseline_v2" / "subset_21_ragas.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
