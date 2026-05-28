"""Compare 1b.3 vs 1.1-b on subset vs full48."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
subset_q = {
    q["query"]
    for q in json.loads(
        Path(r"C:\Users\xsk\Desktop\RAG项目优化\golden_subset_21.json").read_text(
            encoding="utf-8"
        )
    )["test_cases"]
}


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


r3 = load("results/p1b.3-full48/report.json")
r11 = load("results/p1.1-full48/report.json")
r3ref = load("results/p1b.3-full48-ref-cr-cp/report.json")
r3s = load("results/p1b.3-ref-cr-cp/report.json")
r11s = load("results/p1.1-ref-cr-cp/report.json")

by_type: dict = defaultdict(lambda: {"n": 0, "cr3": [], "cr11": [], "in_subset": 0})
for q3, q11 in zip(r3["query_results"], r11["query_results"]):
    t = q3.get("type") or "unknown"
    by_type[t]["n"] += 1
    if q3["query"] in subset_q:
        by_type[t]["in_subset"] += 1
    cr3 = q3.get("metrics", {}).get("context_recall")
    cr11 = q11.get("metrics", {}).get("context_recall")
    if cr3 is not None and cr11 is not None:
        by_type[t]["cr3"].append(cr3)
        by_type[t]["cr11"].append(cr11)

print("=== Full48 CR (old) by type: 1b.3 - 1.1-b ===")
for t, d in sorted(by_type.items(), key=lambda x: -x[1]["n"]):
    if not d["cr3"]:
        continue
    m3 = sum(d["cr3"]) / len(d["cr3"])
    m11 = sum(d["cr11"]) / len(d["cr11"])
    print(f"  {t:14s} n={d['n']} in21={d['in_subset']:2d}  delta={m3-m11:+.3f}")

w3 = w11 = tie = 0
for a, b in zip(r3s["query_results"], r11s["query_results"]):
    sp = 0.6 * a["metrics"]["context_recall"] + 0.4 * a["metrics"]["context_precision"]
    sp2 = 0.6 * b["metrics"]["context_recall"] + 0.4 * b["metrics"]["context_precision"]
    if sp > sp2 + 0.001:
        w3 += 1
    elif sp2 > sp + 0.001:
        w11 += 1
    else:
        tie += 1
print(f"\nSubset21 @ref: 1b.3 wins {w3}, 1.1-b wins {w11}, tie {tie}")

def sp(q: dict) -> float:
    m = q["metrics"]
    return 0.6 * m["context_recall"] + 0.4 * m["context_precision"]


# Same 21q: subset-only run vs full48 run (1b.3)
m_full = {q["query"]: q for q in r3["query_results"]}
m_sub = {q["query"]: q for q in r3s["query_results"]}
d_sp, d_cr = [], []
for q in subset_q:
    if q in m_full and q in m_sub:
        d_sp.append(sp(m_full[q]) - sp(m_sub[q]))
        d_cr.append(
            m_full[q]["metrics"]["context_recall"]
            - m_sub[q]["metrics"]["context_recall"]
        )
print("\n1b.3 同一21题：子集专跑 vs 全量48里同题（旧口径）")
print(f"  ΔScore_p mean: {sum(d_sp)/len(d_sp):+.3f}")
print(f"  ΔCR mean:      {sum(d_cr)/len(d_cr):+.3f}")

in3 = [q for q in r3["query_results"] if q["query"] in subset_q]
out3 = [q for q in r3["query_results"] if q["query"] not in subset_q]
in11 = [q for q in r11["query_results"] if q["query"] in subset_q]
out11 = [q for q in r11["query_results"] if q["query"] not in subset_q]


def mean_cr(qs):
    return sum(q["metrics"]["context_recall"] for q in qs) / len(qs)


in_d = [
    a["metrics"]["context_recall"] - b["metrics"]["context_recall"]
    for a, b in zip(in3, in11)
]
out_d = [
    a["metrics"]["context_recall"] - b["metrics"]["context_recall"]
    for a, b in zip(out3, out11)
]
print("\nFull48 CR delta (1b.3 - 1.1-b, old):")
print(f"  21题子集题 mean: {sum(in_d)/len(in_d):+.3f}")
print(f"  其余27题 mean:   {sum(out_d)/len(out_d):+.3f}")

w3 = w11 = 0
for q in subset_q:
    q3 = next(x for x in r3ref["query_results"] if x["query"] == q)
    q11 = next(x for x in r11["query_results"] if x["query"] == q)
    if sp(q3) > sp(q11) + 0.001:
        w3 += 1
    elif sp(q11) > sp(q3) + 0.001:
        w11 += 1
print(f"\n同一21题、全量48那次跑：@ref(1b.3) vs 旧(1.1-b) 逐题 Score_p：1b.3赢 {w3}，1.1-b赢 {w11}")

# faith delta full
f3 = [q["metrics"].get("faithfulness") for q in r3["query_results"]]
f11 = [q["metrics"].get("faithfulness") for q in r11["query_results"]]
print(f"\nFull48 Faith mean: 1b.3={sum(f3)/len(f3):.3f} 1.1-b={sum(f11)/len(f11):.3f} delta={sum(f3)/len(f3)-sum(f11)/len(f11):+.3f}")
