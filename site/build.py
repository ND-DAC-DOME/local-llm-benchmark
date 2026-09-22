#!/usr/bin/env python3
"""Build the static report site from results/ (run after any new benchmark run).

Writes site/data/summary.json (every number the pages chart) and site/data/items/<task>.json (the question
explorer: prompt, gold, and each model's final answer, verdict, tokens, seconds). Reasoning traces go to
site/data/reasoning/<task>/<id>.json and are fetched only when a reader opens one. LiveCodeBench private tests
are never written.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402

R = ROOT / "results"
OUT = ROOT / "site" / "data"
(OUT / "items").mkdir(parents=True, exist_ok=True)
(OUT / "reasoning").mkdir(parents=True, exist_ok=True)

MODELS = {"muse": "Muse-Glimmer-30B", "qwen36": "Qwen3.6-27B", "qwen38": "Qwen3.8-27B"}
# which run supplies each (model, budget, task)
RUNS = {
    ("muse", "12k"): "muse-glimmer-30b-q4",
    ("qwen36", "12k"): "qwen3.6-27b-q4",
    ("qwen38", "12k"): "qwen3.8-27b-nvfp4-a6000-mtp",
    ("qwen36", "32k"): "qwen3.6-27b-q4-32k",
    ("qwen38", "32k"): "qwen3.8-27b-nvfp4-a6000-mtp-32k",
    ("muse", "32k"): "muse-glimmer-30b-q4-dflash-32k",  # GPQA / LCB only
}
TASKS = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro", "humaneval": "HumanEval",
         "gpqa_diamond": "GPQA Diamond", "livecodebench": "LiveCodeBench v6"}


def load(run, task):
    p = R / run / f"{task}.jsonl"
    if not p.exists():
        return None
    return {str(json.loads(l)["id"]): json.loads(l) for l in p.open()}


def summ(run):
    return json.loads((R / run / "summary.json").read_text())["tasks"]


# ---------------- per-task scores
quality = {}
for task in TASKS:
    quality[task] = {}
    for (model, budget), run in RUNS.items():
        recs = load(run, task)
        if not recs:
            continue
        correct_key = "correct_judge_env" if task == "livecodebench" else "correct"
        n = len(recs)
        c = sum(r[correct_key] for r in recs.values())
        tr = sum(r["finish_reason"] == "length" for r in recs.values())
        wrong = n - c - tr
        fin = [r for r in recs.values() if r["finish_reason"] != "length"]
        quality[task][f"{model}@{budget}"] = {
            "run": run, "n": n, "correct": c, "wrong": wrong, "truncated": tr,
            "accuracy": round(100 * c / n, 1),
            "accuracy_finished": round(100 * sum(r[correct_key] for r in fin) / len(fin), 1) if fin else None,
            "mean_tokens": round(sum(r["completion_tokens"] for r in recs.values()) / n),
            "mean_seconds": round(sum(r["seconds"] for r in recs.values()) / n, 1),
            **({"accuracy_strict": round(100 * sum(r["correct"] for r in recs.values()) / n, 1)} if task == "livecodebench" else {}),
        }

# paired counts on the comparisons the pages show
def paired(task, a, b):
    ra, rb = load(RUNS[a], task), load(RUNS[b], task)
    if not ra or not rb:
        return None
    k = "correct_judge_env" if task == "livecodebench" else "correct"
    ids = ra.keys() & rb.keys()
    return {"both": sum(ra[i][k] and rb[i][k] for i in ids), "only_a": sum(ra[i][k] and not rb[i][k] for i in ids),
            "only_b": sum(rb[i][k] and not ra[i][k] for i in ids), "neither": sum(not ra[i][k] and not rb[i][k] for i in ids),
            "a": f"{a[0]}@{a[1]}", "b": f"{b[0]}@{b[1]}"}

pairs = {
    "mmlu_pro": [paired("mmlu_pro", ("muse", "12k"), ("qwen36", "12k")), paired("mmlu_pro", ("muse", "12k"), ("qwen38", "12k")),
                 paired("mmlu_pro", ("muse", "12k"), ("qwen36", "32k")), paired("mmlu_pro", ("muse", "12k"), ("qwen38", "32k"))],
    "humaneval": [paired("humaneval", ("muse", "12k"), ("qwen36", "12k")), paired("humaneval", ("muse", "12k"), ("qwen38", "12k"))],
    "gpqa_diamond": [paired("gpqa_diamond", ("muse", "32k"), ("qwen38", "32k"))],
    "livecodebench": [paired("livecodebench", ("muse", "32k"), ("qwen38", "32k"))],
}

# ---------------- speed (from summary.json speed tasks)
def speed(run):
    s = summ(run).get("speed")
    if not s:
        return None
    c = s["concurrency_1"]
    return {"decode": c["median_decode_tok_s_per_stream"], "ttft": c["median_ttft_s"],
            "prefill": {k.replace("_tokens", "").replace("~", ""): round(v["median_prefill_tok_s"]) for k, v in s.get("prefill", {}).items()},
            "prefill_ttft": {k.replace("_tokens", "").replace("~", ""): v["median_ttft_s"] for k, v in s.get("prefill", {}).items()}}

speeds = {
    "muse_q4": speed("muse-glimmer-30b-q4"), "muse_q4_dflash": speed("muse-glimmer-30b-q4-dflash"),
    "muse_nvfp4": speed("muse-glimmer-30b-nvfp4-a6000"), "muse_nvfp4_dflash": speed("muse-glimmer-30b-nvfp4-a6000-dflash"),
    "qwen36_q4": speed("qwen3.6-27b-q4"),
    "qwen38_nomtp": speed("qwen3.8-27b-nvfp4-a6000"), "qwen38_mtp": speed("qwen3.8-27b-nvfp4-a6000-mtp"),
    "qwen38_spark": speed("qwen3.8-27b-nvfp4-spark"),
}
# the three Qwen3.8 executions (noise / hardware independence)
qwen38_runs = {}
for label, run in [("A6000, no MTP", "qwen3.8-27b-nvfp4-a6000"), ("A6000, MTP", "qwen3.8-27b-nvfp4-a6000-mtp"), ("Spark, MTP", "qwen3.8-27b-nvfp4-spark")]:
    t = summ(run)
    qwen38_runs[label] = {k: round(100 * t[k]["accuracy"], 1) for k in ["gsm8k", "mmlu_pro", "humaneval"]}
muse_quant = {k: {"Q4 GGUF": round(100 * summ("muse-glimmer-30b-q4")[k]["accuracy"], 1),
                  "NVFP4": round(100 * summ("muse-glimmer-30b-nvfp4-a6000")[k]["accuracy"], 1)} for k in ["gsm8k", "mmlu_pro", "humaneval"]}
mtp_quality = {k: {"no MTP": round(100 * summ("qwen3.8-27b-nvfp4-a6000")[k]["accuracy"], 1),
                   "MTP": round(100 * summ("qwen3.8-27b-nvfp4-a6000-mtp")[k]["accuracy"], 1)} for k in ["gsm8k", "mmlu_pro", "humaneval"]}

# prefix cache (from the saved logs: fixed numbers, documented in results.md section 2b)
prefix_cache = {
    "scenarios": ["Exact repeat, ~6k tokens", "Exact repeat, ~20k tokens", "Shared 6k system prompt, new question", "Multi-turn continuation"],
    "conditions": {
        "Spark, vLLM 0.26.1 + MTP": {"first": [3.05, 16.69, 3.44, 6.18], "second": [3.05, 16.69, 3.42, 3.47], "hits": [0, 0, 0, 0]},
        "A6000, vLLM 0.29 + MTP": {"first": [3.25, 11.58, 3.29, 4.26], "second": [1.60, 1.70, 1.62, 1.64], "hits": [3200, 17600, 3200, 3200]},
        "A6000, vLLM 0.29, no MTP": {"first": [3.20, 11.32, 3.25, 5.05], "second": [0.81, 0.92, 0.82, 0.83], "hits": [4704, 18816, 4704, 4704]},
    },
    "spark_129k": {"first": 164.65, "second": 165.23, "prefill_tok_s": 784, "tokens": 129072},
    "note": "Multi-turn 'first' includes generating 64 tokens; compare the second calls.",
}

(OUT / "summary.json").write_text(json.dumps({
    "models": MODELS, "tasks": TASKS, "runs": {f"{m}@{b}": r for (m, b), r in RUNS.items()},
    "quality": quality, "pairs": pairs, "speeds": speeds, "qwen38_runs": qwen38_runs,
    "muse_quant": muse_quant, "mtp_quality": mtp_quality, "prefix_cache": prefix_cache,
}, indent=1))

# ---------------- question explorer
PROMPT_CACHE = {}
def prompts(task):
    if task not in PROMPT_CACHE:
        docs, _ = bench.TASKS[task](200 if task in ("gsm8k", "mmlu_pro") else 0)
        PROMPT_CACHE[task] = {str(d["id"]): d for d in docs}
    return PROMPT_CACHE[task]

def final_answer(r, task):
    a = bench.strip_think(r["answer"] or "")
    if task in ("humaneval", "livecodebench"):
        return bench.extract_code(a)
    return a

EXPLORER = {  # (task) -> list of (label, model, budget)
    "gsm8k": [("muse", "12k"), ("qwen36", "12k"), ("qwen38", "12k")],
    "mmlu_pro": [("muse", "12k"), ("qwen36", "12k"), ("qwen38", "12k"), ("qwen36", "32k"), ("qwen38", "32k")],
    "humaneval": [("muse", "12k"), ("qwen36", "12k"), ("qwen38", "12k"), ("qwen36", "32k"), ("qwen38", "32k")],
    "gpqa_diamond": [("muse", "32k"), ("qwen38", "32k")],
    "livecodebench": [("muse", "32k"), ("qwen38", "32k")],
}
index = {}
for task, cols in EXPLORER.items():
    pr = prompts(task)
    cols_loaded = {f"{m}@{b}": load(RUNS[(m, b)], task) for m, b in cols}
    ids = list(next(iter(cols_loaded.values())).keys())
    items = []
    (OUT / "reasoning" / task).mkdir(exist_ok=True)
    for i in ids:
        d = pr[i]
        q = d["prompt"]
        # show the question without the harness's formatting instruction
        q = re.sub(r"^(Solve the problem|Answer the multiple choice question|Complete the following Python function|You are an expert Python programmer)[^\n]*\n\n", "", q)
        if task == "livecodebench":
            q = q.split("\n\n### Format\n")[0].replace("### Question\n", "")
        entry = {"id": i, "category": d.get("category"), "gold": str(d.get("gold", "")) if task != "humaneval" else "",
                 "question": q, "models": {}}
        reasoning = {}
        for key, recs in cols_loaded.items():
            r = recs.get(i)
            if not r:
                continue
            ck = "correct_judge_env" if task == "livecodebench" else "correct"
            entry["models"][key] = {"verdict": "truncated" if r["finish_reason"] == "length" else ("correct" if r[ck] else "wrong"),
                                    "tokens": r["completion_tokens"], "seconds": r["seconds"], "answer": final_answer(r, task)[:6000]}
            if r.get("reasoning"):
                reasoning[key] = r["reasoning"]
        (OUT / "reasoning" / task / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', i)}.json").write_text(json.dumps(reasoning))
        items.append(entry)
    (OUT / "items" / f"{task}.json").write_text(json.dumps({"task": task, "columns": [f"{m}@{b}" for m, b in cols], "items": items}))
    index[task] = {"n": len(items), "columns": [f"{m}@{b}" for m, b in cols],
                   "categories": sorted(Counter(e["category"] for e in items if e["category"]).keys())}
(OUT / "index.json").write_text(json.dumps(index, indent=1))
sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in (OUT / "items").glob("*.json")}
print("wrote site/data: summary.json, index.json, items", sizes, "MB; reasoning files:", sum(1 for _ in (OUT / "reasoning").rglob("*.json")))
