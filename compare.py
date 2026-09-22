#!/usr/bin/env python3
"""Print a markdown comparison of every run under results/ (reads summary.json only)."""
import json
import math
from pathlib import Path

runs = {p.parent.name: json.loads(p.read_text())["tasks"]
        for p in sorted(Path(__file__).parent.glob("results/*/summary.json"))}
names = list(runs)
print("| task | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for task in ["gsm8k", "mmlu_pro", "humaneval", "gpqa_diamond", "livecodebench"]:
    cells = []
    for n in names:
        t = runs[n].get(task)
        if not t:
            cells.append("not run")
        elif t["accuracy"] is None:
            cells.append(f"incomplete ({t['errors']} request errors)")
        else:
            p, k = t["accuracy"], t["n"]
            ci = 1.96 * math.sqrt(p * (1 - p) / k)  # normal-approx 95% interval on the sample
            cell = (f"{100 * p:.1f}% ±{100 * ci:.1f} ({t['correct']}/{k}, "
                    f"{t['truncated']} truncated, ~{t['mean_completion_tokens']} tok)")
            if "accuracy_judge_env" in t:  # LiveCodeBench: also the score with numpy/numba available
                cell += f"; judge env {100 * t['accuracy_judge_env']:.1f}%"
            cells.append(cell)
    print(f"| {task} | " + " | ".join(cells) + " |")
cells = []
for n in names:
    s = runs[n].get("speed", {}).get("concurrency_1")
    cells.append(f"{s['median_decode_tok_s_per_stream']} tok/s, TTFT {s['median_ttft_s']}s" if s else "not run")
print("| decode speed (1 stream) | " + " | ".join(cells) + " |")
for size in ["~1024_tokens", "~4096_tokens", "~12288_tokens"]:
    cells = []
    for n in names:
        pf = runs[n].get("speed", {}).get("prefill", {}).get(size)
        cells.append(f"{pf['median_prefill_tok_s']:.0f} tok/s, TTFT {pf['median_ttft_s']}s" if pf else "not run")
    print(f"| prefill {size.replace('_', ' ')} | " + " | ".join(cells) + " |")
