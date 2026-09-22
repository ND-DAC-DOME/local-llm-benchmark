#!/usr/bin/env python3
"""Re-score saved HumanEval answers with the current scorer (no regeneration).

  python rescore_humaneval.py <run-name> [--write]
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

import bench

run, write = sys.argv[1], "--write" in sys.argv
outdir = Path(__file__).parent / "results" / run
ds = {d["task_id"]: d for d in load_dataset("openai/openai_humaneval", split="test")}
recs = [json.loads(l) for l in (outdir / "humaneval.jsonl").open()]
for r in recs:
    new = bool(bench.score_humaneval(bench.strip_think(r["answer"] or ""), ds[r["id"]]))
    if new != r["correct"]:
        print(f"{r['id']}: {r['correct']} -> {new}")
    r["correct"] = new
n_correct = sum(r["correct"] for r in recs)
print(f"{run}: {n_correct}/{len(recs)} = {n_correct / len(recs):.4f}")
if write:
    (outdir / "humaneval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = json.loads((outdir / "summary.json").read_text())
    s["tasks"]["humaneval"].update(correct=n_correct, accuracy=round(n_correct / len(recs), 4),
                                   rescored="prompt helpers kept in scope")
    (outdir / "summary.json").write_text(json.dumps(s, indent=2))
    print("written")
