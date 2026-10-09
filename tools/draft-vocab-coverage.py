#!/usr/bin/env python3
"""How much of the text the models actually generated falls inside the MTP drafter's reduced vocabulary.

The qwen38-flash-next-spark image lets the MTP drafter score only the 65,536 ids listed in
draft_vocab_65536.npy (VLLM_MTP_DRAFT_VOCAB). The target still verifies every drafted token, so
the output is unchanged; the only thing that can drop is the acceptance rate, and it drops exactly
when the target wants an id outside the set. This script measures how often that happens on the
saved benchmark traces: it tokenizes every reasoning trace and answer in results/<run>/<task>.jsonl
with the Flash-Next tokenizer and counts the ids outside the set.

Caveat: the saved traces were written by other models (Qwen3.8-27B, Qwen3.6, Muse), so this is the
token distribution of text of this kind, not of Flash-Next's own output. Re-run it on traces from the
Flash-Next server for the real number.

usage:
  uv run --with tokenizers --with numpy python tools/draft-vocab-coverage.py \
      --ids tools/draft_vocab_65536.npy --tokenizer tools/flash-next-tokenizer [--results results] \
      [--runs 'qwen3.8-*'] [--top 30]
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from tokenizers import Tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    return Tokenizer.from_file(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="draft_vocab_65536.npy from the spark image")
    ap.add_argument("--tokenizer", required=True, help="directory with tokenizer.json, or the file itself")
    ap.add_argument("--results", default="results", help="results root (default: results)")
    ap.add_argument("--runs", default="*", help="glob over run directories (default: all)")
    ap.add_argument("--top", type=int, default=30, help="how many missed tokens to list")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    keep = np.unique(np.load(args.ids).astype(np.int64))
    tok = load_tokenizer(args.tokenizer)
    vocab = tok.get_vocab_size(with_added_tokens=True)
    print(f"draft set: {keep.size:,} ids (min {keep.min()}, max {keep.max()}); tokenizer vocab: {vocab:,}")

    per_cell = {}                  # (run, task) -> [tokens, misses, records, server_tokens]
    missed_all: Counter = Counter()
    missed_by_task: dict[str, Counter] = defaultdict(Counter)

    files = sorted(glob.glob(os.path.join(args.results, args.runs, "*.jsonl")))
    if not files:
        raise SystemExit(f"no jsonl under {args.results}/{args.runs}")

    for path in files:
        run = os.path.basename(os.path.dirname(path))
        task = os.path.splitext(os.path.basename(path))[0]
        texts, server_tokens = [], 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                texts.append((r.get("reasoning") or "") + (r.get("answer") or ""))
                server_tokens += int(r.get("completion_tokens") or 0)
        n_tok = n_miss = 0
        for i in range(0, len(texts), args.batch):
            encs = tok.encode_batch(texts[i : i + args.batch], add_special_tokens=False)
            for e in encs:
                ids = np.asarray(e.ids, dtype=np.int64)
                if ids.size == 0:
                    continue
                out = ids[~np.isin(ids, keep, assume_unique=False)]
                n_tok += ids.size
                n_miss += out.size
                if out.size:
                    c = Counter(out.tolist())
                    missed_all.update(c)
                    missed_by_task[task].update(c)
        per_cell[(run, task)] = [n_tok, n_miss, len(texts), server_tokens]

    # --- per run / task -------------------------------------------------------------------------
    print()
    print(f"{'run':<40} {'task':<14} {'records':>7} {'tokens':>11} {'outside':>9} {'coverage':>9} {'server tok':>11}")
    tot = defaultdict(lambda: [0, 0, 0, 0])
    for (run, task), (n, m, rec, st) in sorted(per_cell.items()):
        cov = 100.0 * (1 - m / n) if n else float("nan")
        print(f"{run:<40} {task:<14} {rec:>7} {n:>11,} {m:>9,} {cov:>8.3f}% {st:>11,}")
        for k in (task, "ALL"):
            t = tot[k]
            t[0] += n; t[1] += m; t[2] += rec; t[3] += st

    print()
    print(f"{'by task':<55} {'records':>7} {'tokens':>11} {'outside':>9} {'coverage':>9}")
    for task, (n, m, rec, _) in sorted(tot.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        cov = 100.0 * (1 - m / n) if n else float("nan")
        print(f"{task:<55} {rec:>7} {n:>11,} {m:>9,} {cov:>8.3f}%")

    # --- what is missed ------------------------------------------------------------------------
    def show(counter: Counter, label: str, k: int) -> None:
        total = sum(counter.values())
        if not total:
            return
        print(f"\n{label}: {total:,} tokens outside the set, {len(counter):,} distinct ids")
        print(f"{'count':>8} {'share':>7}  {'id':>7}  token")
        for tid, cnt in counter.most_common(k):
            print(f"{cnt:>8,} {100.0*cnt/total:>6.2f}%  {tid:>7}  {tok.decode([tid])!r}")

    show(missed_all, "ALL TASKS", args.top)
    for task, c in sorted(missed_by_task.items()):
        show(c, task, max(10, args.top // 3))


if __name__ == "__main__":
    main()
