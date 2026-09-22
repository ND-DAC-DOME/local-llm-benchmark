#!/usr/bin/env python3
"""Re-grade saved LiveCodeBench answers with the current scorer (no regeneration), under two judge environments.

  python rescore_livecodebench.py <run-name> [--write]

  strict  : programs run with the benchmark's own interpreter (standard library only matters)
  judge   : programs run with .venv-judge (numpy, scipy, numba, networkx, sortedcontainers), like AtCoder's real judge;
            see the `judge` group in pyproject.toml. 23 of one model's failures were `import numba`, so both numbers are reported.
Each record keeps `correct` (= strict) and gains `correct_judge_env`; summary.json gains `accuracy_judge_env`.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bench

root = Path(__file__).parent
run, write = sys.argv[1], "--write" in sys.argv
path = root / "results" / run / "livecodebench.jsonl"
recs = [json.loads(l) for l in path.open()]
docs, _ = bench.load_livecodebench(0)
byid = {d["id"]: d for d in docs}


def grade(python):
    bench.LCB_PYTHON = python
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda r: bool(bench.score_livecodebench(bench.strip_think(r["answer"] or ""), byid[r["id"]])), recs))


strict = grade(sys.executable)
judge = grade(str(root / ".venv-judge" / "bin" / "python"))
old = [r["correct"] for r in recs]
n = len(recs)
print(f"{run}: {n} items | as recorded {sum(old)} | strict, fixed scorer {sum(strict)} ({100*sum(strict)/n:.1f}%)"
      f" | judge env {sum(judge)} ({100*sum(judge)/n:.1f}%)")
print("  recorded->strict flips:", sum(a != b for a, b in zip(old, strict)),
      "(to correct:", sum(b and not a for a, b in zip(old, strict)), ", to wrong:", sum(a and not b for a, b in zip(old, strict)), ")")
print("  strict->judge flips   :", sum(a != b for a, b in zip(strict, judge)),
      "(to correct:", sum(b and not a for a, b in zip(strict, judge)), ", to wrong:", sum(a and not b for a, b in zip(strict, judge)), ")")
if write:
    for r, s, j in zip(recs, strict, judge):
        r["correct"], r["correct_judge_env"] = s, j
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    sp = root / "results" / run / "summary.json"
    summ = json.loads(sp.read_text())
    summ["tasks"]["livecodebench"].update(correct=sum(strict), accuracy=round(sum(strict) / n, 4),
                                          correct_judge_env=sum(judge), accuracy_judge_env=round(sum(judge) / n, 4),
                                          rescored="fixed scorer (no injected imports for stdin programs); judge env = .venv-judge")
    sp.write_text(json.dumps(summ, indent=2))
    print("  written")
