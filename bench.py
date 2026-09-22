#!/usr/bin/env python3
"""Quality + speed benchmark for any OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama).

Same prompts, same sampling, same scoring for every model, so runs are comparable.
Every request/response is saved to results/<run>/<task>.jsonl so scores can be audited.

  python bench.py --base-url http://localhost:8001/v1 --model muse-glimmer --run-name muse-q4 \
      --tasks gsm8k,mmlu_pro,humaneval,speed --limit 200 --system "Reasoning strength: high"
"""
import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

SEED = 1234

# Dataset revisions used for the published results (Hugging Face commit SHAs), so a re-run sees the same items.
DATASET_REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "hendrydong/gpqa_diamond_mc": "284143babc24a94fbac45d143333b2307e64ff80",
    "livecodebench/code_generation_lite": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
}


def load_pinned(name, *args, **kwargs):
    return load_dataset(name, *args, revision=DATASET_REVISIONS[name], **kwargs)


# ---------------------------------------------------------------- scoring

def strip_think(text):
    """Drop inline <think> blocks for servers that don't split reasoning_content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_gsm8k(answer, gold):
    m = re.search(r"(?i)final answer\s*[:=]\s*(.+)", answer)
    pred = last_number(m.group(1)) if m else None
    if pred is None:
        pred = last_number(answer)
    return pred is not None and abs(pred - gold) < 1e-6


def score_mmlu_pro(answer, gold):
    m = re.findall(r"(?i:answer)\s*(?:is)?\s*[:=]?\s*[\s*(\[]*([A-J])\b", answer)
    return bool(m) and m[-1].upper() == gold


def extract_code(answer):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", answer, flags=re.S)
    return max(blocks, key=len) if blocks else answer


def score_humaneval(answer, doc):
    code = extract_code(answer)
    if f"def {doc['entry_point']}" not in code:
        # Body-only reply: complete the original signature.
        code = doc["prompt"] + code
    else:
        # Full-function reply: keep the prompt's imports and helper functions (e.g. is_palindrome,
        # poly) in scope; the model's later definition of the entry point overrides the stub.
        code = doc["prompt"] + "\n    pass\n\n" + code
    program = f"{code}\n\n{doc['test']}\n\ncheck({doc['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(f.name).unlink(missing_ok=True)


def score_gpqa(answer, gold):
    """Last \\boxed{X} or 'Answer: X' with X in A-D."""
    m = re.findall(r"\\boxed\{\s*\(?([A-D])\)?\s*\}|(?i:answer)\s*(?:is)?\s*[:=]?\s*[\s*(\[]*([A-D])\b", answer)
    letters = [a or b for a, b in m]
    return bool(letters) and letters[-1] == gold


# Header for call-based (LeetCode-style) problems only, where the model writes just `class Solution`.
# `from builtins import *` comes last so star-imports cannot shadow builtins (math.pow would break pow(a, b, mod)).
# stdin/stdout programs are run exactly as written, with no header.
LCB_IMPORTS = ("import sys, math, collections, heapq, bisect, itertools, functools, re, string, random, json\n"
               "from typing import *\nfrom collections import *\nfrom functools import *\nfrom itertools import *\n"
               "from heapq import *\nfrom bisect import *\nfrom math import *\nfrom builtins import *\n")
# Interpreter used to execute model-written programs. Default: this one (standard library + whatever the venv has).
# Point LCB_PYTHON at an environment with numpy/scipy/numba/... to mimic a real judge such as AtCoder's.
LCB_PYTHON = os.environ.get("LCB_PYTHON") or sys.executable
LCB_DRIVER = """
import json as _json, sys as _sys
_tests = _json.load(open(_sys.argv[1]))
_sol = Solution()
for _t in _tests:
    _args = [_json.loads(_l) for _l in _t["input"].split("\\n")]
    _out = getattr(_sol, _sys.argv[2])(*_args)
    if isinstance(_out, tuple):
        _out = list(_out)
    if _out != _json.loads(_t["output"]):
        _sys.exit(1)
"""


def _same_output(got, want):
    g = [l.strip() for l in got.strip().splitlines()]
    w = [l.strip() for l in want.strip().splitlines()]
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):  # token-wise, tolerant to float formatting
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x != y:
                try:
                    if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                        return False
                except ValueError:
                    return False
    return True


def score_livecodebench(answer, doc, per_test_timeout=6):
    """Run the model's program against every public + private test; pass = all tests pass."""
    code = extract_code(answer)
    if "```" not in answer and not code.strip():
        return False
    tests = doc["tests"]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sol.py"
        if doc["func_name"]:  # LeetCode style: call Solution().<func>(*args)
            src.write_text(LCB_IMPORTS + code + "\n" + LCB_DRIVER)
            tf = Path(td) / "tests.json"
            tf.write_text(json.dumps(tests))
            try:
                r = subprocess.run([LCB_PYTHON, str(src), str(tf), doc["func_name"]], capture_output=True,
                                   timeout=min(300, per_test_timeout * len(tests)), cwd=td)
                return r.returncode == 0
            except subprocess.TimeoutExpired:
                return False
        src.write_text(code + "\n")  # stdin/stdout style: run the program exactly as written
        for t in tests:
            try:
                r = subprocess.run([LCB_PYTHON, str(src)], input=t["input"], capture_output=True, text=True,
                                   timeout=per_test_timeout, cwd=td)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0 or not _same_output(r.stdout, t["output"]):
                return False
    return True


# ---------------------------------------------------------------- tasks

def load_gsm8k(limit):
    ds = load_pinned("openai/gsm8k", "main", split="test")
    docs = [{"id": i, "prompt": ("Solve the problem. End your reply with a line of the form "
                                 "'Final answer: <number>'.\n\n" + d["question"]),
             "gold": float(d["answer"].split("####")[-1].strip().replace(",", ""))}
            for i, d in enumerate(ds)]
    return sample(docs, limit), lambda a, d: score_gsm8k(a, d["gold"])


def load_mmlu_pro(limit):
    ds = load_pinned("TIGER-Lab/MMLU-Pro", split="test")
    docs = []
    for d in ds:
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(d["options"]))
        docs.append({"id": d["question_id"], "category": d["category"], "gold": d["answer"],
                     "prompt": ("Answer the multiple choice question. End your reply with a line of "
                                "the form 'Answer: <letter>'.\n\n" + d["question"] + "\n" + opts)})
    return sample(docs, limit), lambda a, d: score_mmlu_pro(a, d["gold"])


def load_humaneval(limit):
    ds = load_pinned("openai/openai_humaneval", split="test")
    docs = [{"id": d["task_id"], "entry_point": d["entry_point"], "test": d["test"],
             "prompt_src": d["prompt"],
             "prompt": ("Complete the following Python function. Reply with the full function "
                        "in a single ```python code block.\n\n" + d["prompt"])}
            for d in ds]
    return sample(docs, limit), lambda a, d: score_humaneval(a, {**d, "prompt": d["prompt_src"]})


def load_gpqa_diamond(limit):
    ds = load_pinned("hendrydong/gpqa_diamond_mc", split="test")  # ungated mirror of GPQA Diamond, 198 items
    docs = []
    for i, d in enumerate(ds):
        q = d["problem"].split("Please write your final answer")[0].rstrip()
        docs.append({"id": i, "category": d["domain"], "gold": re.search(r"([A-D])", d["solution"]).group(1),
                     "prompt": ("Answer the multiple choice question. End your reply with a line of the form "
                                "'Answer: <letter>'.\n\n" + q)})
    return sample(docs, limit), lambda a, d: score_gpqa(a, d["gold"])


def load_livecodebench(limit):
    """LiveCodeBench code generation, release v6 slice (175 problems, contests 2025-01-04..2025-04-06)."""
    import base64
    import pickle
    import zlib
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset",
                           revision=DATASET_REVISIONS["livecodebench/code_generation_lite"])
    docs = []
    for line in open(path):
        d = json.loads(line)
        tests = json.loads(d["public_test_cases"]) + json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(d["private_test_cases"].encode()))))
        func = json.loads(d["metadata"] or "{}").get("func_name", "")
        if func:
            fmt = ("You will use the following starter code to write the solution and enclose your code in a "
                   "single ```python code block.\n```python\n" + d["starter_code"] + "\n```")
        else:
            fmt = ("Read the inputs from stdin, solve the problem, and write the answer to stdout. "
                   "Enclose your complete program in a single ```python code block.")
        docs.append({"id": d["question_id"], "category": f"{d['platform']}/{d['difficulty']}", "func_name": func,
                     "tests": tests, "prompt": ("You are an expert Python programmer. Solve the problem below; "
                                                "your program must pass all tests.\n\n### Question\n"
                                                + d["question_content"] + "\n\n### Format\n" + fmt)})
    return sample(docs, limit), lambda a, d: score_livecodebench(a, d)


def sample(docs, limit):
    if limit and limit < len(docs):
        docs = random.Random(SEED).sample(docs, limit)
    return docs


TASKS = {"gsm8k": load_gsm8k, "mmlu_pro": load_mmlu_pro, "humaneval": load_humaneval,
         "gpqa_diamond": load_gpqa_diamond, "livecodebench": load_livecodebench}


# ---------------------------------------------------------------- runner

def chat(client, args, prompt, max_tokens=None, stream=False):
    msgs = ([{"role": "system", "content": args.system}] if args.system else [])
    msgs.append({"role": "user", "content": prompt})
    return client.chat.completions.create(
        model=args.model, messages=msgs, temperature=args.temperature, top_p=args.top_p,
        max_tokens=max_tokens or args.max_tokens, seed=SEED, stream=stream,
        **({"stream_options": {"include_usage": True}} if stream else {}))


def run_task(client, args, name, outdir):
    docs, scorer = TASKS[name](args.limit)
    out = outdir / f"{name}.jsonl"
    done = {}
    if out.exists():  # resume
        for line in out.open():
            r = json.loads(line)
            done[str(r["id"])] = r
    if args.retry_truncated_from:  # keep everything from the source run except the truncated items
        src = Path(__file__).parent / "results" / args.retry_truncated_from / f"{name}.jsonl"
        for line in src.open():
            r = json.loads(line)
            if r.get("finish_reason") != "length" and str(r["id"]) not in done:
                r["copied_from"] = args.retry_truncated_from
                done[str(r["id"])] = r
                f_copy = out.open("a"); f_copy.write(json.dumps(r) + "\n"); f_copy.close()
    todo = [d for d in docs if str(d["id"]) not in done]

    def one(doc):
        t0 = time.time()
        try:
            resp = chat(client, args, doc["prompt"])
            msg = resp.choices[0].message
            answer = strip_think(msg.content or "")
            rec = {"id": doc["id"], "correct": bool(scorer(answer, doc)), "answer": msg.content,
                   # llama.cpp returns reasoning_content, vLLM returns reasoning
                   "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
                   "finish_reason": resp.choices[0].finish_reason,
                   "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                   "seconds": round(time.time() - t0, 2), "gold": doc.get("gold"),
                   "category": doc.get("category")}
        except Exception as e:  # noqa: BLE001 - recorded, counted separately, never scored
            rec = {"id": doc["id"], "error": repr(e)}
        return rec

    streak = 0
    with out.open("a") as f, ThreadPoolExecutor(args.concurrency) as pool:
        for rec in tqdm(pool.map(one, todo), total=len(todo), desc=name):
            if "error" not in rec:  # errors are not persisted so a resume retries them
                f.write(json.dumps(rec) + "\n")
                f.flush()
                streak = 0
            else:
                streak += 1
                with (outdir / "errors.log").open("a") as ef:
                    ef.write(f"{time.strftime('%H:%M:%S')} {name} {rec['id']} {rec['error']}\n")
                if streak >= 5:  # server is gone; stop instead of burning through the task list
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(f"{name}: 5 consecutive request errors, aborting (see errors.log)")
            done[str(rec["id"])] = rec

    recs = [done[str(d["id"])] for d in docs]
    ok = [r for r in recs if "error" not in r]
    errors = len(recs) - len(ok)
    truncated = sum(r["finish_reason"] == "length" for r in ok)
    n_correct = sum(r["correct"] for r in ok)
    toks = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    return {"n": len(recs), "scored": len(ok), "correct": n_correct,
            # request errors say nothing about the model: no accuracy until a re-run fills them in
            "accuracy": round(n_correct / len(recs), 4) if recs and not errors else None,
            "errors": errors, "truncated": truncated,
            "mean_completion_tokens": round(statistics.mean(toks)) if toks else None,
            "mean_seconds": round(statistics.mean(r["seconds"] for r in ok), 1) if ok else None}


SPEED_PROMPT = "Write a detailed, 600-word explanation of how a CPU cache hierarchy works."


def run_speed(client, args, outdir):
    """Decode tok/s and time-to-first-token, single stream and under concurrency."""
    def one(_):
        t0 = time.time()
        first, usage = None, None
        for chunk in chat(client, args, SPEED_PROMPT, max_tokens=512, stream=True):
            if chunk.usage:
                usage = chunk.usage
            if first is None and chunk.choices:
                d = chunk.choices[0].delta
                if (d.content or getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)):
                    first = time.time()
        end = time.time()
        n = usage.completion_tokens if usage else None
        return {"ttft_s": round(first - t0, 3) if first else None, "completion_tokens": n,
                "decode_tok_s": round(n / (end - first), 2) if n and first else None,
                "total_s": round(end - t0, 2)}

    results = {}
    for conc in [int(c) for c in args.speed_concurrency.split(",")]:
        t0 = time.time()
        with ThreadPoolExecutor(conc) as pool:
            rows = list(pool.map(one, range(max(conc, 3))))
        wall = time.time() - t0
        total = sum(r["completion_tokens"] or 0 for r in rows)
        ttft = [r["ttft_s"] for r in rows if r["ttft_s"]]
        decode = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
        results[f"concurrency_{conc}"] = {
            "requests": len(rows),
            # None (not a crash) when first-token detection failed; rows keep the raw data
            "median_ttft_s": statistics.median(ttft) if ttft else None,
            "median_decode_tok_s_per_stream": statistics.median(decode) if decode else None,
            "aggregate_tok_s": round(total / wall, 2), "rows": rows}
    results["prefill"] = run_prefill(client, args)
    (outdir / "speed.json").write_text(json.dumps(results, indent=2))
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()}
    summary["prefill"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                          for k, v in results["prefill"].items()}
    return summary


def run_prefill(client, args):
    """Prompt-processing speed: a long prompt, max_tokens=1, prompt_tokens / time to first token.

    Servers cache prompt prefixes (vLLM --enable-prefix-caching, llama.cpp slot cache), so every
    request starts with a unique nonce and the filler is rotated; otherwise repeats look instant.
    """
    filler = " ".join(d["question"] for d in load_pinned("openai/gsm8k", "main", split="test"))
    out = {}
    for target in [int(t) for t in args.prefill_tokens.split(",")]:
        rows = []
        for rep in range(3):
            nonce = f"session {random.randrange(10**9)} rep {rep} target {target}. "
            start = (rep * 7919 * target) % max(1, len(filler) - target * 4)
            text = nonce + filler[start:start + target * 4]  # ~4 chars per token
            prompt = text + "\n\nReply with the single word OK."
            t0 = time.time()
            first = None
            for chunk in chat(client, args, prompt, max_tokens=1, stream=True):
                if first is None and chunk.choices:
                    d = chunk.choices[0]
                    if (d.delta.content or getattr(d.delta, "reasoning_content", None)
                            or getattr(d.delta, "reasoning", None) or d.finish_reason):
                        first = time.time()
            usage = getattr(chunk, "usage", None)
            n = usage.prompt_tokens if usage else None
            rows.append({"prompt_tokens": n, "ttft_s": round(first - t0, 3) if first else None,
                         "prefill_tok_s": round(n / (first - t0), 1) if n and first else None})
        ok = [r for r in rows if r["prefill_tok_s"]]
        out[f"~{target}_tokens"] = {
            "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in ok) if ok else None,
            "median_ttft_s": statistics.median(r["ttft_s"] for r in ok) if ok else None,
            "median_prefill_tok_s": statistics.median(r["prefill_tok_s"] for r in ok) if ok else None,
            "rows": rows}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True, help="model id as the server exposes it")
    p.add_argument("--run-name", required=True)
    p.add_argument("--api-key", default="none")
    p.add_argument("--tasks", default="gsm8k,mmlu_pro,humaneval,speed")
    p.add_argument("--limit", type=int, default=200, help="items per task (0 = full set)")
    p.add_argument("--system", default="", help="e.g. 'Reasoning strength: high'")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--retry-truncated-from", default="",
                   help="run name: re-send only the items that hit the token limit in that run's saved results "
                        "(e.g. the same run with a larger --max-tokens); other items are copied over unchanged")
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel requests; only raise on a server you own (some llama.cpp models abort under parallel slots)")
    p.add_argument("--speed-concurrency", default="1")
    p.add_argument("--prefill-tokens", default="1024,4096,12288",
                   help="approximate prompt lengths for the prefill measurement")
    args = p.parse_args()

    outdir = Path(__file__).parent / "results" / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800)

    summary = {"config": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": {}}
    if (outdir / "summary.json").exists():  # re-running one task keeps the others' results
        summary["tasks"] = json.loads((outdir / "summary.json").read_text())["tasks"]
    for name in args.tasks.split(","):
        summary["tasks"][name] = (run_speed(client, args, outdir) if name == "speed"
                                  else run_task(client, args, name, outdir))
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(name, json.dumps(summary["tasks"][name]))


if __name__ == "__main__":
    main()
