#!/usr/bin/env python
"""Merge @ref CR/CP with faith (adjusted) from source run into *-ref-cr-cp report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FORMULA = "0.4*CR + 0.3*Faith + 0.3*CP"


def score_full(cr: float, cp: float, faith: float) -> float:
    return round(0.4 * cr + 0.3 * faith + 0.3 * cp, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True, help="Run with faith scored (e.g. p1.5-full48-v34)")
    p.add_argument("--ref-dir", required=True, help="@ref CR/CP dir (e.g. p1.5-full48-v34-ref-cr-cp)")
    p.add_argument(
        "--faith-key",
        default="faith_adjusted_boundary5_weida",
        help="Key under faith_adjusted block (default: boundary5_weida)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    ref_dir = Path(args.ref_dir)
    src_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    ref_report = json.loads((ref_dir / "report.json").read_text(encoding="utf-8"))

    agg_ref = ref_report["aggregate_metrics"]
    cr = float(agg_ref["context_recall"])
    cp = float(agg_ref["context_precision"])
    faith_raw = float(src_report.get("faithfulness_mean") or src_report["aggregate_metrics"]["faithfulness"])
    adj_block = src_report.get("faith_adjusted") or {}
    faith_adj = float(adj_block.get(args.faith_key, faith_raw))

    sf = score_full(cr, cp, faith_adj)
    ref_report["faithfulness"] = faith_adj
    ref_report["faithfulness_raw"] = faith_raw
    ref_report["faith_source"] = str(source)
    ref_report["faith_kind"] = adj_block.get("method", args.faith_key)
    ref_report["scoring_note_faith"] = adj_block.get("method", "")
    ref_report["score_full"] = sf
    ref_report["score_full_formula"] = FORMULA
    agg_ref["faithfulness"] = faith_adj
    ref_report["aggregate_metrics"] = agg_ref

    (ref_dir / "report.json").write_text(json.dumps(ref_report, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = ref_dir / "rescore_meta.json"
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "faithfulness": faith_adj,
            "faithfulness_raw": faith_raw,
            "faith_kind": ref_report["faith_kind"],
            "score_full": sf,
            "score_full_formula": FORMULA,
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Patched {ref_dir}")
    print(f"  CR={cr} CP={cp} Faith={faith_adj} (raw={faith_raw}) Score={sf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
