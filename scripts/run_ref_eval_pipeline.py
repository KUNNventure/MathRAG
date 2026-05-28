#!/usr/bin/env python
"""End-to-end: retrieve+generate → @ref CR/CP → Faith → Faith adjust → Score.

Does NOT run legacy (ground_truth) CR/CP during evaluate.

Usage:
    python scripts/run_ref_eval_pipeline.py \\
        --results-dir results/p1.5-full48-v34 \\
        --test-set "C:\\Users\\xsk\\Desktop\\RAG项目优化\\golden_test_set.json" \\
        --no-rerank --prompt-path config/prompts/answer_generation_v3.4.txt

    # Or post-process existing run (skip step 1):
    python scripts/run_ref_eval_pipeline.py --source-dir results/p1.5-full48-v34 --skip-retrieve
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_TEST_SET = r"C:\Users\xsk\Desktop\RAG项目优化\golden_test_set.json"
DEFAULT_RESULTS = "results/p1.5-full48-v34"
DEFAULT_PROMPT = "config/prompts/answer_generation_v3.4.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="@ref CR/CP + adjusted Faith pipeline")
    p.add_argument("--source-dir", default=None, help="Run dir (required if --skip-retrieve)")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS, help=f"Output dir (default: {DEFAULT_RESULTS})")
    p.add_argument("--skip-retrieve", action="store_true", help="Only rescore + faith + patch")
    p.add_argument("--ref-dir", default=None, help="@ref output (default: <source>-ref-cr-cp)")
    p.add_argument("--python", default=None, help="Python executable")
    p.add_argument("--test-set", default=DEFAULT_TEST_SET)
    p.add_argument("--no-rerank", action="store_true", default=True)
    p.add_argument("--with-rerank", action="store_true", help="Enable reranker (overrides default --no-rerank)")
    p.add_argument("--prompt-path", default=DEFAULT_PROMPT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--collection", default=None)
    p.add_argument("evaluate_args", nargs=argparse.REMAINDER, help="Extra args for evaluate.py")
    return p.parse_args()


def py_exe(args: argparse.Namespace) -> str:
    if args.python:
        return args.python
    for cand in (PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe", PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"):
        if cand.is_file():
            return str(cand)
    return sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    py = py_exe(args)

    source = Path(args.source_dir or args.results_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else Path(str(source) + "-ref-cr-cp")

    if not args.skip_retrieve:
        results_dir = str(source)
        ev_cmd = [
            py,
            "scripts/evaluate.py",
            "--no-metrics",
            "--results-dir",
            results_dir,
            "--test-set",
            args.test_set,
            "--prompt-path",
            args.prompt_path,
            "--workers",
            str(args.workers),
        ]
        if args.collection:
            ev_cmd.extend(["--collection", args.collection])
        if not args.with_rerank:
            ev_cmd.append("--no-rerank")
        extra = [a for a in args.evaluate_args if a != "--"]
        ev_cmd.extend(extra)
        run(ev_cmd)
        source = Path(results_dir)

    if not (source / "report.json").is_file():
        print(f"Missing {source / 'report.json'}", file=sys.stderr)
        return 1

    run([py, "scripts/rescore_ragas_cr_cp.py", "--source-dir", str(source), "--out-dir", str(ref_dir)])
    run([py, "scripts/score_faith_from_report.py", "--source-dir", str(source)])
    run([py, "scripts/apply_ref_faith_patch.py", "--source-dir", str(source), "--ref-dir", str(ref_dir)])

    print("\n✅ Pipeline complete:")
    print(f"   retrieve+gen: {source}/")
    print(f"   @ref CR/CP:   {ref_dir}/")
    print(f"   Faith+Score:  see {ref_dir}/report.json (score_full)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
