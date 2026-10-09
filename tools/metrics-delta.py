#!/usr/bin/env python3
"""Speculative-decoding acceptance and prefix-cache hit rate of one benchmark run, from two /metrics snapshots.

vLLM's counters are cumulative since boot. Snapshot them before and after a run (curl .../metrics | grep vllm:)
and the difference is what that run alone did:

  uv run python tools/metrics-delta.py results/<run>/metrics-before.txt results/<run>/metrics-after.txt

With a single snapshot the figures are cumulative since the server booted (every user's traffic).
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^(vllm:\w+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$')


def load(path: str) -> dict:
    out = defaultdict(float)
    for line in open(path):
        m = LINE.match(line.strip())
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] += value
    return out


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    a = load(sys.argv[1]) if len(sys.argv) == 3 else {}
    b = load(sys.argv[-1])
    d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}

    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    drafted = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    gen = d.get("vllm:generation_tokens_total", 0)
    prompt = d.get("vllm:prompt_tokens_total", 0)
    pc_q = d.get("vllm:prefix_cache_queries_total", 0)
    pc_h = d.get("vllm:prefix_cache_hits_total", 0)

    print(f"{'generated tokens':<34} {gen:>12,.0f}")
    print(f"{'prompt tokens':<34} {prompt:>12,.0f}")
    print(f"{'draft steps':<34} {drafts:>12,.0f}")
    print(f"{'drafted tokens':<34} {drafted:>12,.0f}")
    print(f"{'accepted tokens':<34} {accepted:>12,.0f}")
    if drafted:
        print(f"{'acceptance per drafted token':<34} {100*accepted/drafted:>11.1f}%")
    if drafts:
        print(f"{'tokens per step (accepted + 1)':<34} {accepted/drafts + 1:>12.2f}")
        pos = sorted((int(k[k.index('[')+1:-1]), v) for k, v in d.items()
                     if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total["))
        for p, v in pos:
            print(f"{'  accepted at position ' + str(p + 1):<34} {100*v/drafts:>11.1f}%")
    if pc_q:
        print(f"{'prefix cache hit rate':<34} {100*pc_h/pc_q:>11.1f}%  ({pc_h:,.0f} of {pc_q:,.0f} queried tokens)")


if __name__ == "__main__":
    main()
