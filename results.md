# Muse-Glimmer-30B vs Qwen3.6-27B vs Qwen3.8-27B — results

Runs on 2026-09-17..22 (all complete) on a workstation with 2× RTX A6000 48 GB and, for one run, on the team's DGX Spark over the network. Same 200 questions (GSM8K, MMLU-Pro; seed 1234), all 164 HumanEval problems, same prompts, temperature 1.0 / top_p 0.95, 12,288-token completion budget unless stated. A truncated answer counts as wrong. Every wrong answer of every run was inspected; there were no request errors and no answer-parsing failures in any run. Reproduction: `README.md`, `runs/`.

## 1. Quality

| | Muse Q4 GGUF | Muse NVFP4 | Qwen3.6 Q4 GGUF | Qwen3.8 NVFP4 | Qwen3.8 NVFP4 + MTP | Qwen3.8 NVFP4, Spark |
|---|---|---|---|---|---|---|
| Runtime | llama.cpp | vLLM | llama.cpp | vLLM, no MTP | vLLM, MTP | vLLM, MTP |
| GSM8K (200) | 97.0% | 97.0% | 96.0% | 98.0% | 97.5% | 98.5% |
| MMLU-Pro (200) | **82.0%** | **81.0%** | 73.0% | 74.5% | 77.0% | 76.0% |
| HumanEval (164) | **98.8%** | 97.6% | 93.3% | 94.5% | 97.0% | 95.7% |
| Truncated at 12k: GSM8K / MMLU-Pro / HumanEval | 0 / 3 / 1 | 0 / 4 / 1 | 5 / 19 / 7 | 0 / 27 / 8 | 1 / 24 / 5 | 0 / 30 / 7 |
| Mean tokens/answer: GSM8K / MMLU-Pro / HumanEval | 570 / 1,592 / 1,181 | 553 / 1,590 / 1,347 | 1,993 / 3,244 / 3,298 | 449 / 3,159 / 1,833 | 521 / 3,109 / 1,438 | 398 / 3,397 / 1,671 |

95% margin of error: GSM8K ±2–3, MMLU-Pro ±5–6, HumanEval ±2–4 points. Empirically, the same Qwen3.8 checkpoint run three times (A6000 no MTP, A6000 MTP, Spark) spread by ±1.5 points on MMLU-Pro and ±1.3 on HumanEval; treat differences of that size as noise.

**Token budget experiment (MMLU-Pro, 200 items, 32,768-token budget):**

| | Qwen3.6 Q4 GGUF, 32k | Qwen3.8 NVFP4 + MTP, 32k | Muse Q4 GGUF, 12k (reference) |
|---|---|---|---|
| MMLU-Pro accuracy | **82.0%** (164/200) | **80.0%** (160/200) | 82.0% (164/200) |
| MMLU-Pro truncated | 1 | 7 | 3 |
| MMLU-Pro mean tokens/answer | 4,083 | 4,935 | 1,592 |
| MMLU-Pro mean seconds/item (3 in flight) | 176 | 98 | 50 |
| GSM8K | 97.0% (0 truncated) | 98.5% (0 truncated) | 97.0% |
| HumanEval | 97.0% (159/164, 0 truncated) | 97.6% (160/164, 1 truncated) | 98.8% |

At 12k → 32k on MMLU-Pro: Qwen3.6 73.0% → 82.0% (16 of its 19 truncated items became correct; net +18), Qwen3.8 74.5% → 80.0% (13 of 27; net +11). HumanEval also rises with the larger budget (Qwen3.6 93.3% → 97.0%, Qwen3.8 97.0% → 97.6%), which puts both Qwens level with Muse on every task once they are allowed to finish thinking. Per question against Muse (12k): Qwen3.6-32k 7 vs 7, Qwen3.8-32k 7 vs 11 — both statistical ties. Qwen3.6-32k vs Qwen3.8-32k: 9 vs 5, a tie.

**Per-question comparisons (same items, 12k budget)**

| Pair | Only first right | Only second right | Both right |
|---|---|---|---|
| Muse Q4 vs Qwen3.6 Q4, MMLU-Pro | 22 | 4 | 142 |
| Muse Q4 vs Qwen3.8 NVFP4 (no MTP), MMLU-Pro | 21 | 6 | 143 |
| **Muse NVFP4 vs Qwen3.8 NVFP4 (same vLLM), MMLU-Pro** | 17 | 4 | — |
| Muse Q4 vs Qwen3.6 Q4, HumanEval | 10 | 1 | 152 |
| Muse Q4 vs Qwen3.8 NVFP4 (no MTP), HumanEval | 7 | 0 | 155 |
| Muse NVFP4 vs Qwen3.8 NVFP4 + MTP, HumanEval | 3 | 2 | — |
| Muse Q4 vs Muse NVFP4, MMLU-Pro | 5 | 3 | 159 |
| Qwen3.8 12k vs Qwen3.8 32k (MTP), MMLU-Pro | 3 | 14 | — |

**Reading**

- GSM8K is saturated: every configuration lands at 96–98.5%.
- Under a 12k-token budget, Muse-Glimmer is ahead of both Qwens on MMLU-Pro (4–9 points, paired counts 17–22 vs 4–6) and ahead of Qwen3.6 on HumanEval; against Qwen3.8 with MTP the HumanEval difference is noise.
- The Qwen gap is mostly *not finishing*: Qwen3.6/3.8 run out of 12k tokens on 19–30 MMLU-Pro items vs 3–4 for Muse. Among items each model completed, MMLU-Pro accuracy is Muse 83%, Qwen3.6 81%, Qwen3.8 86%. Given 32k tokens, both Qwens tie Muse on MMLU-Pro (82.0% / 80.0% vs 82.0%), at 2.6–3.1× the tokens per answer and 2–3.5× the time per item.
- Quantization did not change Muse: NVFP4 (NVIDIA AutoQuant) and unsloth Q4 GGUF agree on every task within noise. This also answers the comparability concern: Muse vs Qwen3.8 both in NVFP4 on the same vLLM shows the same MMLU-Pro gap as the GGUF runs.
- MTP speculative decoding did not change Qwen3.8's scores (as expected; it is lossless) and halved its time per item.
- Same checkpoint on the Spark and on the A6000 gives the same scores.
- Cost: Muse uses about half the tokens per answer of either Qwen on MMLU-Pro and HumanEval.

## 1b. Harder tasks: GPQA Diamond and LiveCodeBench v6 (run 8)

GSM8K and HumanEval turned out to be near the ceiling for these models (96–99%), so they work as sanity checks rather than as a ranking. Run 8 adds two tasks with headroom, at a 32,768-token budget, each model in its fastest lossless configuration (Muse Q4 GGUF + DFlash on llama.cpp, 2 requests in flight; Qwen3.8 NVFP4 + MTP on vLLM, 3 in flight). Qwen3.6 was left out by decision.

| | Muse-Glimmer-30B | Qwen3.8-27B |
|---|---|---|
| **GPQA Diamond** (198), accuracy | **80.3%** (159) | **79.8%** (158) |
| GPQA: wrong answer / truncated at 32k | 38 / 1 | 9 / 31 |
| GPQA: accuracy among finished items | 80.7% | 94.6% |
| GPQA: Physics / Chemistry / Biology | 82/86, 65/93, 12/19 | 82/86, 63/93, 13/19 |
| GPQA: mean tokens / seconds per item | 5,180 / 130 s | 11,811 / 236 s |
| **LiveCodeBench v6** (175), judge env (numpy/numba available, like AtCoder) | **78.9%** (138) | **60.0%** (105) |
| LiveCodeBench, strict (standard library only) | 66.9% (117) | 60.0% (105) |
| LCB: truncated at 32k | 19 | **66** |
| LCB: accuracy among finished items | 88% | 96% |
| LCB: easy / medium / hard (judge env) | 43/43, 46/52, 49/80 | 43/43, 41/52, 21/80 |
| LCB: mean tokens / seconds per item | 10,574 / 265 s | 17,948 / 381 s |

Per question: GPQA both right 145, only Muse 14, only Qwen 13 (a tie). LiveCodeBench both 104, **only Muse 34**, only Qwen 1; of Qwen's 66 truncated problems Muse solved 34.

**Reading**

- GPQA Diamond is a tie on the score, with opposite error profiles: Muse finishes almost everything and is wrong on 38; Qwen3.8 is right 95% of the time when it finishes but does not finish 31 questions (16%) even with 32k tokens. Both scores sit close to the vendors' published numbers for these models at full precision (Meta: 83.5 for Muse; the gap to Qwen's 89.2 is the budget).
- LiveCodeBench is the first task where the two separate clearly, and it is the same mechanism at a larger scale: Qwen3.8 runs out of 32k tokens on 38% of the problems (56 of the 80 hard ones). Given a budget it finishes in, it is very accurate (96%); it just needs far more than 32k on hard competitive-programming problems. Its vendor number (90.3, with up to 80k tokens) is consistent with that.
- Cost: Qwen3.8 used 2.3× the tokens on GPQA and 1.7× on LiveCodeBench, and 1.4–1.8× the time per item, for equal or lower scores.
- **Judge environment matters for LiveCodeBench.** 23 of Muse's strict failures were solutions that `import numba` (a JIT compiler that AtCoder's judge provides); with the same libraries available, 21 of them pass, all on hard problems. Qwen3.8 did not use such libraries, so its score is unchanged. Both numbers are reported; the judge-env score is the headline because it matches how the benchmark is meant to be run. A scorer bug was also fixed before re-grading (an injected `from math import *` shadowed the built-in `pow(a, b, mod)`; stdin programs now run exactly as written). Every saved answer was re-graded with the fixed scorer, under both environments, without regenerating anything (`rescore_livecodebench.py`).

**Speed during real work (from the per-item records: tokens generated / seconds, per request, with 2–3 requests in flight)**

| | Muse Q4 + DFlash | Qwen3.8 NVFP4 + MTP |
|---|---|---|
| GPQA Diamond, median per request | 40.0 tok/s | 50.5 tok/s |
| LiveCodeBench, median per request | 40.0 tok/s | 47.1 tok/s |
| Idle-server decode, single request (same servers, measured after the tasks) | 56.1 tok/s | 52.7 tok/s |
| Draft acceptance during real work | DFlash 52–69% of drafted tokens (≈2.6–3.1 per step; highest on code) | MTP 56%, ≈2.7 tokens per pass |

Per-request speed under load is lower than the idle single-stream number because requests share the GPU; Qwen3.8 ran 3 in flight and Muse 2 (Muse is a dense transformer with a larger KV cache), so the two columns are not exactly like for like. On an idle server Muse + DFlash is the faster decoder; under 3-way load Qwen3.8 + MTP gets more per request.

## 2. Speed (one request at a time, server otherwise idle)

| | Muse Q4 GGUF, llama.cpp | Muse NVFP4, vLLM | Qwen3.6 Q4 GGUF, llama.cpp | Qwen3.8 NVFP4, vLLM, no MTP | Qwen3.8 NVFP4, vLLM, MTP | Qwen3.8 NVFP4, Spark (MTP) |
|---|---|---|---|---|---|---|
| Decode, tok/s (no spec. decoding / with) | 38.5 / **56.5** (DFlash) | 28.3 / 24.7 (DFlash) | 33.7 / — | 31.8 | **50.8** (MTP) | 20.3 (MTP) |
| TTFT, short prompt | 0.19 s | 0.23 s | 0.17 s | 0.16 s | 0.20 s | 0.25 s |
| Prefill ~1k tokens, tok/s | 855 | 1,542 | 887 | 1,764 | 1,743 | **2,189** |
| Prefill ~4k tokens, tok/s | 1,288 | 1,579 | 1,165 | 1,845 | 1,836 | 1,846 |
| Prefill ~12k tokens, tok/s | 1,425 | 1,557 | 1,183 | 1,791 | 1,789 | 1,355 |
| TTFT with a ~12k-token prompt | 8.5 s | 7.7 s | 10.6 s | 7.0 s | 7.0 s | 9.2 s |

**Speculative decoding on and off (same model, same runtime, same GPU; quality is unaffected by speculative decoding)**

| | Without | With | Change | Drafter / acceptance |
|---|---|---|---|---|
| Qwen3.8 NVFP4, vLLM, decode | 31.8 tok/s | **50.8 tok/s** | +60% | MTP head, 3 tokens |
| Muse Q4 GGUF, llama.cpp, decode | 38.5 tok/s | **56.5 tok/s** | +47% | unsloth `dflash-kquant.gguf`; 40–44% of drafted tokens accepted, ~2.2 per step |
| Muse NVFP4, vLLM, decode | 28.3 tok/s | 24.7 tok/s (15 draft tokens), 22.0 (5) | −13% / −22% | `meta-models/Muse-Glimmer-30B-assistant`; only 25% of drafted tokens accepted, 1.3 per step |
| Qwen3.6 Q4 GGUF, llama.cpp | 33.7 tok/s | not possible | — | the unsloth GGUF has no MTP layers |

Prefill is unchanged by speculative decoding (Muse GGUF + DFlash: 718–732 tok/s vs 855–1,425 without; the drafter's own prefill adds a little). With DFlash, Muse Q4 on llama.cpp is the fastest configuration measured on the A6000 (56.5 tok/s), ahead of Qwen3.8 NVFP4 + MTP on vLLM (50.8). The DFlash result on vLLM is specific to this Ampere setup (Marlin FP8/FP4 emulation, Inductor off); on the hardware NVIDIA tested (Blackwell, SGLang) it may well help.

**Reading**

- **A6000 vs Spark, same model, same runtime, same MTP setting:** decode 50.8 vs 20.3 tok/s (2.5×); prefill equal at 4k, Spark ahead on 1k, A6000 ahead on 12k. For a 40,000-character (~10k-token) system prompt, both start answering in about 6–9 s.
- MTP gives Qwen3.8 +60% decode speed on the A6000 (31.8 → 50.8 tok/s) with no quality change.
- Muse on vLLM is slower than Muse on llama.cpp (28 vs 38.5 tok/s without speculative decoding). The vLLM run carries two Ampere workarounds that cost speed (Inductor off, FP8 layers through Marlin), and DFlash does not help it there (table above). Like-for-like with speculative decoding on both: Muse Q4 + DFlash on llama.cpp 56.5 tok/s vs Qwen3.8 NVFP4 + MTP on vLLM 50.8 tok/s.
- Prefill is mostly a runtime property here: vLLM ≈1.5–1.8k tok/s, llama.cpp ≈0.9–1.4k tok/s for similar-size models.
- Every Spark figure below is reproducible from the commands listed in `docs/spark-measurements.md`. Spark speed numbers are from its live shared server; two measurements a day apart agreed (20.5 / 20.3 tok/s). Its vLLM (0.26.1 dev, 2026-08-26) uses the native `FlashInferCutlassNvFp4LinearKernel` and FlashInfer attention (verified in its serve log), and its MTP accepts 52.6% of drafted tokens (70% / 51% / 37% by position), i.e. ~2.6 tokens emitted per model pass (from its `/metrics`). At 20.3 tok/s that is ~127 ms per pass, consistent with a memory-bandwidth-bound GB10: a decode pass reads ~19 GB (MLPs 10.6 GB, of which layers 0-55 are NVFP4, 8.4 GB, and layers 56-63 are FP8, 2.1 GB; all attention projections and the `lm_head` in FP8, 8.5 GB) plus ~6 GB for the three MTP draft steps (BF16 MTP head + FP8 `lm_head` each), i.e. ~25 GB per step, ~93 ms at the nominal 273 GB/s.

## 2b. Prefix caching with MTP: the Spark vs vLLM 0.29 (run 9)

The Spark's operator observed that speculative decoding seemed to disable prefix caching on this model. Verified with the latency tests in `tools/` (same prompt sent twice, `max_tokens=1`, server idle): on the Spark (vLLM 0.26.1 dev + MTP) the repeat is never faster and adds zero cache hits, for exact repeats of 6k, 20k and 129k tokens, a shared 6k system prompt, and a multi-turn continuation. The same five scenarios on vLLM 0.29.0 (A6000, same recipe):

| Second request, same prefix | Spark, 0.26.1 + MTP | A6000, 0.29 + MTP | A6000, 0.29, no MTP |
|---|---|---|---|
| Exact repeat, ~6k tokens | 3.05 s (= first) | 1.60 s (first 3.25 s) | 0.81 s (first 3.20 s) |
| Exact repeat, ~20k tokens | 16.7 s (= first) | 1.70 s (first 11.6 s) | 0.92 s (first 11.3 s) |
| Shared ~6k system prompt, new question | 3.42 s (= first) | 1.62 s | 0.82 s |
| Multi-turn continuation | 3.47 s (= first) | 1.64 s | 0.83 s |
| Cache hits on the repeat | 0 | multiples of 1,600 tokens | multiples of 224 tokens |

So it is the vLLM version, not MTP: 0.29 caches with MTP on (2× faster repeats; it keeps the cache in 1,600-token blocks under speculative decoding on this hybrid model, "dense checkpointing") and caches better without it (4×, 224-token blocks). A vLLM upgrade on the Spark should restore prefix caching with MTP kept on. Caveat: the Spark runs `max_num_seqs 4` and a 262k context, the A6000 test used 3 and 131k. Log: `results/run_09_prefix_cache.log`; reproduce with `runs/09_prefix_cache_vllm029.sh`.

## 3. Failure audit

- Every wrong answer was classified as truncated / wrong answer / parser miss / harness error. Parser misses: 0 in all runs. Harness error: the 4 Muse HumanEval "failures" in the very first pass (prompt helper functions not in scope) — fixed, re-scored, and every later run used the fixed scorer.
- HumanEval failures that are not truncations are genuine unit-test failures on well-formed code (Muse Q4: 1; Muse NVFP4: 3; Qwen3.6: 4; Qwen3.8 no MTP: 1; Qwen3.8 MTP: 4 of 5 non-truncated; Spark: see jsonl).

## 4. Setup and deviations

- Weights: `unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL`, `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL`, `unsloth/Qwen3.8-27B-NVFP4` @ `f0b7c9e` (the Spark serves snapshot `57926ba`; the only commits between them touch README.md, weights identical since the 2026-08-15 squash), `nvidia/Muse-Glimmer-30B-NVFP4` @ `4781837`. Runtimes: llama.cpp `server-cuda` (Sept 2026), vLLM 0.29.0.
- Muse: system prompt `Reasoning strength: high`, top_k 64. Qwens: default thinking mode, top_k 20. Each vendor's recommendation.
- Quantization comparability: `UD-Q4_K_XL` is a dynamic quant (per-layer allocation differs by model). `nvidia/Muse-Glimmer-30B-NVFP4` is ModelOpt AutoQuant (mixed NVFP4/FP8/BF16 chosen by search). `unsloth/Qwen3.8-27B-NVFP4` is mixed precision (tensor dtypes read from the safetensors headers): only the MLPs of layers 0-55 are NVFP4 (8.4 GB); the MLPs of layers 56-63 (2.1 GB), all linear-attention and full-attention projections and the `lm_head` (8.5 GB) are FP8; embeddings, MTP head and vision encoder are BF16. Confirmed independently by a teammate from the same headers. None of the pairs is bit-for-bit the same scheme; the Muse GGUF-vs-NVFP4 agreement suggests it does not matter for these tasks.
- A6000 deviations from the Spark recipe (`docs/spark-qwen3.8-27b-recipe.yaml`): x86 vLLM image; NVFP4 via Marlin weight-only emulation (no FP4 tensor cores on Ampere); `patches/humming_utils.py`; 131k context instead of 262k. The "no MTP" run additionally had no vision encoder, 32k context and 4k prefill batches because GPU 0 was shared at the time.
- Muse NVFP4 on the A6000: `--linear-backend marlin`, Inductor compilation disabled, CUDA graphs on. See `runs/04_vllm_muse_nvfp4.sh`.
- Per-item data: `results/<run>/<task>.jsonl`; table: `.venv/bin/python compare.py`.

## 5. Caveats

- Our prompts and parsers; scores compare these models under identical conditions and are not comparable to public leaderboards.
- No agentic / SWE-bench-style task. Vendor SWE-bench numbers: `swe-bench-cards.md`. LiveCodeBench (run 8) is single-shot competitive programming, not an agentic workflow.
- All weights are 4-bit (or mixed 4/8-bit); vendor model-card numbers are presumably full precision.
- Speed on the A6000 was measured with the other GPU busy with a different run for part of the time; CPU contention is possible but decode speed is GPU-bound.
