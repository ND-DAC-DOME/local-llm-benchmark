# Local LLM benchmark: Muse-Glimmer-30B vs Qwen3.6-27B vs Qwen3.8-27B

A small, auditable test bed for comparing locally served models under identical conditions: same questions, same prompts, same sampling, same token budget, with every prompt, reasoning trace, answer and verdict saved to disk.

**Report site:** https://nd-dac-dome.github.io/local-llm-benchmark/ (the findings, navigable, with a question explorer). Same content as text in [`results.md`](results.md); source in [`site/`](site/). What the datasets are and how they are graded: [`benchmarks.md`](benchmarks.md). Vendor-published SWE-bench numbers (not reproduced by us): [`swe-bench-cards.md`](swe-bench-cards.md).

## Layout

| File | Purpose |
|---|---|
| `bench.py` | The harness. Talks to any OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama). Tasks: `gsm8k`, `mmlu_pro`, `humaneval`, `gpqa_diamond`, `livecodebench`, `speed` (decode tok/s, TTFT, prefill tok/s). Resumable; errors logged; aborts after 5 consecutive request errors. |
| `compare.py` | Prints a markdown table across every run under `results/`, with margins of error. |
| `rescore_humaneval.py`, `rescore_livecodebench.py` | Re-grade saved answers with the current scorer, without regenerating (used after scorer fixes; the LiveCodeBench one also grades under the judge environment). |
| `run_all.sh` | Run 1 end to end: serve Muse then Qwen3.6 (GGUF, llama.cpp) and run all tasks. |
| `speed_only.sh` | Re-measure only speed/prefill for an existing llama.cpp run, keeping its scores. |
| `serve_vllm_nvfp4.sh` | Serve an NVFP4 checkpoint with vLLM on an A6000; parameterized by env vars (model, GPU, port, MTP, ...). |
| `patches/humming_utils.py`, `.patch`, `README.md` | 3-line fix for a vLLM 0.29.0 crash on the FP8 `lm_head` of `unsloth/Qwen3.8-27B-NVFP4` in the Marlin (Ampere) path. Bind-mounted over the installed file. |
| `runs/01..09_*.sh` | One script per configuration that was actually run. This is the reproduction entry point. |
| `config.env` | Shared settings: HF cache path, container images pinned by digest, Spark URL. Override in `config.local.env` (git-ignored). |
| `pyproject.toml`, `uv.lock` | Python dependencies, pinned; the `judge` group is the LiveCodeBench judge environment. |
| `tools/check_prefix_cache.py`, `tools/prefix_cache_scenarios.py`, `tools/spark_inspect.sh` | Latency-based prefix-cache checks for any vLLM server (robust to other users; five usage patterns; `--long` for the 129k-token test) and a read-only inspection of a vLLM server's `/metrics`. Used on the Spark and in run 9. |
| `docs/spark-measurements.md` | Provenance of every Spark number in `results.md`: the exact command behind each one. |
| `MODELS.md`, `verify_models.sh` | Model repositories, snapshots and SHA-256 of the GGUF files; script to verify a local cache against them. |
| `docs/spark-qwen3.8-27b-recipe.yaml` | The DGX Spark's serving recipe that runs 2, 3 and 5 are derived from. Verified on 2026-09-19 against the live `vllm serve` process on the Spark (identical arguments; 70.6 GB of unified memory at `--gpu-memory-utilization 0.55`). No other model server and no Muse-Glimmer weights exist on the Spark. The Spark's snapshot is `57926ba` (weights identical to `f0b7c9e`), its vLLM is 0.26.1 dev with native NVFP4 kernels. |
| `results/<run>/` | `summary.json` (scores + config), `<task>.jsonl` (one line per item: prompt id, reasoning, answer, verdict, tokens, seconds), `speed.json`, `errors.log` if any. |

## Configurations that were run

| # | Run name (under `results/`) | Model / weights | Runtime | Hardware | Spec. decoding |
|---|---|---|---|---|---|
| 1 | `muse-glimmer-30b-q4` | `unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL` | llama.cpp `server-cuda` | A6000 | none |
| 1 | `qwen3.6-27b-q4` | `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` | llama.cpp `server-cuda` | A6000 | none |
| 2 | `qwen3.8-27b-nvfp4-a6000` | `unsloth/Qwen3.8-27B-NVFP4` @ `f0b7c9e` | vLLM 0.29.0 | A6000 | **no MTP** (VRAM-limited config) |
| 3 | `qwen3.8-27b-nvfp4-a6000-mtp` (all tasks, 12k budget), `...-mtp-32k` (all tasks, 32k budget) | same | vLLM 0.29.0 | A6000, whole GPU | **MTP, 3 tokens** (Spark recipe) |
| 4 | `muse-glimmer-30b-nvfp4-a6000` | `nvidia/Muse-Glimmer-30B-NVFP4` @ `4781837` | vLLM 0.29.0 | A6000 | none (NVIDIA vLLM recipe) |
| 5 | `qwen3.8-27b-nvfp4-spark` | `unsloth/Qwen3.8-27B-NVFP4` | vLLM (Spark image) | DGX Spark, over the network | MTP, 3 tokens |
| 6 | `qwen3.6-27b-q4-32k` | `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL` | llama.cpp | A6000 | none; all tasks with a 32k budget |
| 8 | `qwen3.8-27b-nvfp4-a6000-mtp-32k`, `muse-glimmer-30b-q4-dflash-32k` | as runs 3 and 7 | vLLM / llama.cpp | A6000 | MTP / DFlash. **GPQA Diamond (198) and LiveCodeBench v6 (175)** at a 32k budget, then speed on the same servers. AIME was dropped (vendors report 94–95%: no headroom); Qwen3.6 left out by decision. |
| 7 | `muse-glimmer-30b-q4-dflash`, `muse-glimmer-30b-nvfp4-a6000-dflash` (15 draft tokens), `...-dflash5` (5) | as runs 1, 4 | llama.cpp / vLLM | A6000 | **speed only**, with speculative decoding: DFlash (unsloth `dflash-kquant.gguf` for llama.cpp, `meta-models/Muse-Glimmer-30B-assistant` for vLLM). Qwen3.6 `draft-mtp` was attempted and cannot run: the unsloth GGUF has no MTP layers. |

Common settings: temperature 1.0, top_p 0.95 (both vendors' recommendation), seed 1234, 200 fixed items for GSM8K and MMLU-Pro (seed 1234), all 164 HumanEval, 12,288 completion tokens unless the run name says `32k`, 3 requests in flight. Muse gets the system prompt `Reasoning strength: high` and top_k 64; Qwens run in their default thinking mode with top_k 20. Reasoning is returned separately by the server and is never scored.

## Reproducing

```bash
# 1. Python environments with uv (exact versions from uv.lock)
uv sync                                                          # harness -> .venv
UV_PROJECT_ENVIRONMENT=.venv-judge uv sync --group judge         # LiveCodeBench judge env (numpy, numba, ...)
# add a dependency: uv add <pkg>   (or uv add --group judge <pkg> for the judge env); never uv pip install
# 2. Machine settings: copy and edit. HF_CACHE = where model weights are cached; SPARK_URL only for runs/05.
printf 'HF_CACHE=/path/to/huggingface/cache\nSPARK_URL=http://<spark-host>:18300/v1\n' > config.local.env
# 3. Container images are pinned by digest in config.env (llama.cpp build 11011, vLLM 0.29.0); docker pulls them on first use.
# 4. After the first GGUF download, check the files are the ones used here:
./verify_models.sh
./runs/01_gguf_muse_vs_qwen3.6.sh        # ~10 h: GGUF head-to-head
./runs/02_vllm_qwen3.8_nvfp4_no_mtp.sh   # ~4 h
GPU=1 ./runs/03_vllm_qwen3.8_nvfp4_mtp.sh   # needs a free 48 GB GPU
./runs/04_vllm_muse_nvfp4.sh             # ~4 h
SPARK=http://<spark>:18300/v1 ./runs/05_spark_qwen3.8.sh
./runs/06_gguf_qwen3.6_mmlu_32k.sh
./runs/07_speculative_speed.sh          # needs both GPUs free
./runs/08_hard_tasks.sh                 # GPQA Diamond + LiveCodeBench v6 at 32k, then speed; needs both GPUs free
./runs/09_prefix_cache_vllm029.sh       # prefix-cache check, vLLM 0.29 + MTP on/off (~20 min, one GPU)
.venv/bin/python compare.py
```

Every script serves the model in a Docker container with a host-RAM cap (8 GB for llama.cpp, 16 GB for vLLM), runs `bench.py`, and removes the container. `bench.py` resumes from the saved `.jsonl` files, so an interrupted run can simply be restarted. Model weights are pulled from Hugging Face into `$HF_CACHE` (see `config.env`); exact snapshots and file hashes are in [`MODELS.md`](MODELS.md).

To benchmark any other OpenAI-compatible server:

```bash
.venv/bin/python bench.py --base-url http://host:port/v1 --model <served name> --run-name <name> \
  --limit 200 --temperature 1.0 --top-p 0.95 --max-tokens 12288 --concurrency 3 [--system "..."]
```

## How things are measured

- **GSM8K / MMLU-Pro:** zero-shot, one chat turn per item. The prompt asks for a final line `Final answer: <number>` / `Answer: <letter>`; the last such line is compared with the gold answer. An answer that hits the token limit without a final line counts as wrong.
- **HumanEval:** the model returns the function in a code block; the harness prepends the original prompt (imports, helper functions, signature) and executes the problem's unit tests in a subprocess with a 20 s timeout. Pass = all tests pass.
- **Decode speed / TTFT:** one short prompt, `max_tokens=512`, streaming, 3 sequential requests; `completion_tokens / (last chunk − first chunk)`; median.
- **Prefill:** prompts of ~1k, ~4k and ~12k tokens (GSM8K text), `max_tokens=1`, streaming; `prompt_tokens / time to first token`; 3 repetitions each with a unique nonce so prefix caching cannot serve a repeat; median.
- **GPQA Diamond:** 198 four-option questions (ungated mirror `hendrydong/gpqa_diamond_mc`), graded on the last `Answer: <letter>` or `\\boxed{<letter>}`.
- **LiveCodeBench v6:** the 175 problems of `test6.jsonl` (contests 2025-01-04 to 2025-04-06; 112 AtCoder stdin/stdout, 63 LeetCode call-based). The program must pass every public and private test (6 s per test); output compared line by line, tolerant to float formatting. stdin programs run exactly as written; call-based ones get a typing/collections header. Two scores are reported: **strict** (programs executed with the benchmark's own interpreter, standard library only) and **judge env** (executed with `.venv-judge`: numpy, scipy, numba, networkx, sortedcontainers, i.e. what a real AtCoder judge offers; the `judge` group in `pyproject.toml`). The judge-env score is the headline; 23 of Muse's strict failures were solutions that `import numba`. `rescore_livecodebench.py` re-grades saved answers under both without regenerating anything.
- Token counts come from the server's `usage` field, never estimated.
- Every wrong answer of every run was inspected (truncated vs. wrong vs. parser miss); see `results.md`.

## Report site

`site/` is a static site (plain HTML + JavaScript, Chart.js from a CDN) with the findings organised for two readers: a summary and question explorer for deciding, and speed / method / caveats / all-numbers pages for checking. It reads JSON generated from `results/`:

```bash
.venv/bin/python site/build.py      # regenerate site/data/ after any new run (never writes LiveCodeBench private tests)
python3 -m http.server -d site 8000  # preview at http://localhost:8000
```

To publish on GitHub Pages: repository Settings → Pages → "Deploy from a branch", branch `main`, folder `/site`. The explorer fetches `site/data/` files relative to the page, so it works from Pages, from a local server, and from any static host.

## What "reproducible" means here

- **Pinned:** container images (digest), Python packages (`uv.lock`), dataset revisions (`DATASET_REVISIONS` in `bench.py`), model snapshots (`MODELS.md`), item samples (seed 1234; the pinned loaders were checked to return exactly the item ids of the saved runs).
- **Not deterministic:** sampling uses temperature 1.0 as both vendors recommend, and GPU kernels are not bit-reproducible across hardware. Expect scores to move by about ±1.5 points between identical runs (measured: the same Qwen3.8 checkpoint scored 74.5 / 77.0 / 76.0 on MMLU-Pro in three runs). Compare against the margins in `results.md`, not digit by digit.
- **Hardware:** one 48 GB GPU per model (two to run pairs in parallel), Docker with the NVIDIA runtime, ~120 GB of disk for weights. Ampere-specific workarounds are documented in `serve_vllm_nvfp4.sh` and `runs/04`, `runs/07`; on Blackwell they should not be needed.
- **Audit trail:** every prompt id, reasoning trace, answer, verdict, token count and latency is in `results/<run>/<task>.jsonl`.

## Known deviations from the Spark recipe on the A6000 (all hardware-driven)

- x86 `vllm/vllm-openai:latest` instead of the Spark's ARM64 image; no `--load-format instanttensor`.
- NVFP4 GEMMs run through vLLM's Marlin weight-only FP4 emulation (Ampere has no FP4 tensor cores). Same weights, different kernel; affects speed, not answers.
- `patches/humming_utils.py` (see file header).
- Run 2 only: no MTP, no vision encoder, 32k context, 4k prefill batches, because GPU 0 was shared at the time. Run 3 restores MTP, 16k batches and a 131k context on a free GPU.
- Muse NVFP4: `--linear-backend marlin` (Ampere cannot execute the checkpoint's FP8 W8A8 layers) and Inductor compilation disabled (crashes on this model); CUDA graphs kept.

## Caveats

- Scores use our own prompts and parsers and are for comparing models under identical conditions, not against public leaderboards.
- `UD-Q4_K_XL` is a dynamic quant (per-layer bit allocation differs between models); `nvidia/Muse-Glimmer-30B-NVFP4` is ModelOpt AutoQuant (mixed NVFP4/FP8/BF16 chosen by search); `unsloth/Qwen3.8-27B-NVFP4` is mixed precision: NVFP4 MLPs in layers 0-55 (8.4 GB), FP8 MLPs in layers 56-63 (2.1 GB), FP8 attention projections and `lm_head` (8.5 GB), BF16 embeddings / MTP head / vision encoder. None of the pairs is bit-for-bit "the same quantization".
- Speed depends on runtime; compare speed only within the same runtime, or the same model+runtime across machines.
- No agentic / SWE-bench-style task yet.
