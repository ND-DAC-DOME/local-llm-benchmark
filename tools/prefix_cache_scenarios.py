#!/usr/bin/env python3
"""Which usage patterns does a vLLM server's prefix cache actually help? Five light scenarios, latency-based.

  python3 prefix_cache_scenarios.py [--url http://localhost:18300] [--model qwen3.8-27b]

For each scenario the second request shares a long prefix with the first. If the cache works for that pattern,
the second request is much faster and the hit counter grows by about the shared prefix length. The hit counter
is global (other users move it too); the latency of your own request is the reliable signal.
Standard library only.
"""
import argparse, json, random, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:18300")
ap.add_argument("--model", default="qwen3.8-27b")
a = ap.parse_args()

def hits():
    t = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    m = re.search(r"^vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.e+]+)", t, re.M)
    return float(m.group(1)) if m else float("nan")

def ask(messages, max_tokens=1):
    b = json.dumps({"model": a.model, "max_tokens": max_tokens, "temperature": 0, "messages": messages}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        a.url + "/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=900))
    return time.time() - t0, r["usage"]["prompt_tokens"], r["choices"][0]["message"].get("content") or ""

def filler(n, tag):
    return f"session {random.randrange(10**12)} {tag}. " + "The quick brown fox jumps over the lazy dog. " * n

def pair(label, first, second, first_max_tokens=1):
    h0 = hits(); c, n1, _ = ask(first, first_max_tokens); h1 = hits(); w, n2, _ = ask(second, 1); h2 = hits()
    print(f"{label:55s} first {c:6.2f}s ({n1} tok) | second {w:6.2f}s ({n2} tok) | hits +{h1-h0:.0f} / +{h2-h1:.0f}")

p = filler(600, "A") + "\nReply OK."
pair("A) exact repeat, ~6k tokens", [{"role": "user", "content": p}], [{"role": "user", "content": p}])
p = filler(600, "B") + "\nReply OK."
pair("B) exact repeat, first call generates 64 tokens", [{"role": "user", "content": p}], [{"role": "user", "content": p}], 64)
p = filler(2000, "C") + "\nReply OK."
pair("C) exact repeat, ~20k tokens (chunked prefill)", [{"role": "user", "content": p}], [{"role": "user", "content": p}])
s = filler(600, "D")
pair("D) same ~6k system prompt, different question",
     [{"role": "system", "content": s}, {"role": "user", "content": "What is 2+2? One word."}],
     [{"role": "system", "content": s}, {"role": "user", "content": "Name a colour. One word."}])
p = filler(600, "E") + "\nReply OK."
t, n, reply = ask([{"role": "user", "content": p}], 64)
h1 = hits()
w, n2, _ = ask([{"role": "user", "content": p}, {"role": "assistant", "content": reply or "OK"},
                {"role": "user", "content": "Thanks. Reply OK again."}], 1)
print(f"{'E) multi-turn continuation of a ~6k conversation':55s} first {t:6.2f}s ({n} tok) | second {w:6.2f}s ({n2} tok) | hits +{hits()-h1:.0f}")
