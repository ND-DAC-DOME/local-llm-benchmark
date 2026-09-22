#!/usr/bin/env python3
"""Check whether a vLLM server's prefix cache is working, robustly to other users on the same server.

  python3 check_prefix_cache.py [--url http://localhost:18300] [--model qwen3.8-27b] [--pairs 5]

Method: for each pair, send a NEW long prompt (cold), then the SAME prompt again (warm), with max_tokens=1.
  - Primary signal: per-request latency. It belongs to your request only, so other users cannot fake it.
    A working cache makes the warm call several times faster than the cold one.
  - Control: every pair uses a fresh random nonce, so "warm is faster" cannot be a one-off warm-up effect.
  - Other users: the server's running/waiting request count is read before each pair; busy pairs are flagged
    and left out of the verdict. Global hit counters are printed only as a secondary, noisy signal.
Standard library only. Load: 2 short requests per pair.
"""
import argparse, json, random, re, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:18300")
ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--pairs", type=int, default=5)
ap.add_argument("--repeat", type=int, default=600, help="filler sentences; 600 ~ 6k tokens (must exceed the cache block size)")
ap.add_argument("--long", action="store_true", help="one pair with a ~129k-token prompt (12900 sentences), as run on the Spark on 2026-09-21; occupies the server for ~6 min")
a = ap.parse_args()
if a.long:
    a.repeat, a.pairs = 12900, 1

def metric(name):
    txt = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    m = re.search(r"^vllm:" + name + r"\{[^}]*\}\s+([\d.e+]+)", txt, re.M)
    return float(m.group(1)) if m else None

def ask(prompt):
    body = json.dumps({"model": a.model, "max_tokens": 1, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    u = r.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")  # only if the server reports it
    return time.time() - t0, u.get("prompt_tokens"), cached

rows = []
for i in range(a.pairs):
    busy = (metric("num_requests_running") or 0) + (metric("num_requests_waiting") or 0)
    prompt = (f"session {random.randrange(10**12)} pair {i}. "
              + "The quick brown fox jumps over the lazy dog. " * a.repeat + "\nReply OK.")
    h0 = metric("prefix_cache_hits_total")
    cold, n, c_cold = ask(prompt)
    h1 = metric("prefix_cache_hits_total")
    warm, _, c_warm = ask(prompt)
    h2 = metric("prefix_cache_hits_total")
    rows.append(dict(busy=busy, cold=cold, warm=warm, n=n, c_warm=c_warm,
                     d_cold=(h1 - h0) if None not in (h0, h1) else None,
                     d_warm=(h2 - h1) if None not in (h1, h2) else None))
    flag = "  <- server was busy, excluded" if busy else ""
    extra = f" | server-reported cached tokens on warm call: {c_warm}" if c_warm is not None else ""
    print(f"pair {i+1}: {n} prompt tokens | cold {cold:6.2f}s | warm {warm:6.2f}s | speed-up {cold/warm:4.1f}x"
          f" | global hit counter +{rows[-1]['d_cold']:.0f} / +{rows[-1]['d_warm']:.0f}{extra}{flag}")
    time.sleep(1)

clean = [r for r in rows if not r["busy"]] or rows
cold, warm = statistics.median(r["cold"] for r in clean), statistics.median(r["warm"] for r in clean)
print(f"\nclean pairs used: {len(clean)}/{len(rows)} | median cold {cold:.2f}s | median warm {warm:.2f}s | speed-up {cold/warm:.1f}x")
if warm < 0.5 * cold:
    print("VERDICT: prefix cache is WORKING (repeating a prompt is at least 2x faster, consistently across fresh prompts).")
elif warm < 0.8 * cold:
    print("VERDICT: inconclusive (some speed-up). Re-run with more --pairs when the server is idle.")
else:
    print("VERDICT: prefix cache is NOT being used for these prompts (repeating a prompt is not faster).")
print("Note: the global hit counter also counts other users' requests; trust the latency figures above.")
