# How every Spark number in `results.md` was obtained

Nothing was installed or changed on the Spark. If the server requires a key, export `SPARK_API_KEY` (or put it in `.env`); every command below sends it. All measurements are network requests to its vLLM server
(`SPARK_URL`, port 18300) or read-only shell commands run there by a team member. Dates are 2026-09-18..21.

| Number in the report | How it was measured | Reproduce with |
|---|---|---|
| GSM8K / MMLU-Pro / HumanEval scores, decode 20.3 tok/s, prefill 2,189 / 1,846 / 1,355 tok/s | Full benchmark over the network, 12k budget, 3 requests in flight; speed measured after the tasks | `SPARK_URL=... ./runs/05_spark_qwen3.8.sh` → `results/qwen3.8-27b-nvfp4-spark/` |
| Decode 20.5 tok/s (first measurement, a day earlier) | Speed task only | `bench.py --base-url $SPARK_URL --model qwen3.8-27b --tasks speed --speed-concurrency 1 --temperature 1.0 --top-p 0.95` |
| MTP accepts 53% of drafted tokens (70 / 51 / 37% by position), ~2.6 tokens per pass; prefix-cache counter 86.7% | Prometheus counters of the live server (`/metrics`), cumulative since its start | `tools/spark_inspect.sh $SPARK_URL` |
| Server idle / busy at a given moment | `num_requests_running`, `num_requests_waiting`, 20 s deltas of the token counters | `tools/spark_inspect.sh` (section "load right now"); watch with `watch -n 5 ...` |
| Prefix cache has no effect for prompts of ~6k tokens (exact repeat, shared system prompt, multi-turn) and ~20k tokens | Same prompt sent twice, `max_tokens=1`, latency and hit counter compared; server idle | `tools/prefix_cache_scenarios.py --url $SPARK_URL` (5 scenarios) and `tools/check_prefix_cache.py --url $SPARK_URL --pairs 3` |
| Prefix cache has no effect at ~129k tokens: 164.7 s cold vs 165.2 s repeated, 784 tok/s prefill, 0 of 129,072 tokens hit | One cold/warm pair with a 12,900-sentence prompt, idle server checked before each call | `tools/check_prefix_cache.py --url $SPARK_URL --long` (occupies the server ~6 min) |
| Native FP4 kernel (`FlashInferCutlassNvFp4LinearKernel`), FlashInfer attention, vLLM 0.26.1 dev, `enable_prefix_caching=True` | vLLM startup log inside the sparkrun container | on the Spark: `docker exec <sparkrun container> sh -c 'grep -iE "NvFp4LinearKernel\|Marlin\|native support for FP4\|attention backend\|Initializing a V1" /tmp/sparkrun_serve.log'` |
| Serve arguments identical to `docs/spark-qwen3.8-27b-recipe.yaml`; 70.6 GB of unified memory used; 40 GB available; no other model server, no Muse weights, no llama-swap | Read-only shell commands on the Spark | `ps -eo pid,user,args \| grep 'vllm serve'`, `nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv`, `free -g`, `docker ps -a`, `sudo find / -maxdepth 7 -iname '*glimmer*' -o -iname '*muse*'` |
| Checkpoint snapshot on the Spark is `57926ba` (weights identical to `f0b7c9e`) | Model path in the vLLM startup log; commit list of the Hugging Face repo | log line above; `curl https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/commits/main` |
| Bytes read per decode pass (~19 GB model + ~6 GB MTP steps) | Tensor dtypes and shapes from the safetensors headers of the local copy of the same checkpoint | `python3 -c` snippet in `MODELS.md` ("How the checkpoint mix was determined") |

| Same five scenarios cache correctly on vLLM 0.29 (A6000), with and without MTP | Run 9 on the A6000, same recipe as the Spark's | `./runs/09_prefix_cache_vllm029.sh` → `results/run_09_prefix_cache.log` |

## Run 10: the Spark's new model, Qwen3.8-Flash-Next (2026-10-09)

Endpoint `$SPARK_URL` over HTTPS with a private CA and a bearer key (`. ./env.sh` loads `SPARK_URL`, `SPARK_API_KEY`, `SPARK_CA_BUNDLE`). Model id `qwen3.8-flash-next`.

| Number in the report | How it was measured | Reproduce with |
|---|---|---|
| Decode 36.2 tok/s, TTFT 0.22 s; prefill 1,546 / 2,150 / 2,594 tok/s | Speed task, single request, server verified idle (`running-watch.txt`) | `./runs/10_spark_flash_next.sh` → `results/qwen3.8-flash-next-spark-idle/speed.json` |
| MTP acceptance 62.2% per drafted token, 2.24 tokens per step, 71.7% / 52.6% by position | Difference of two `/metrics` snapshots taken around that run (7,044 drafted tokens) | `uv run python tools/metrics-delta.py results/qwen3.8-flash-next-spark-idle/metrics-before.txt results/qwen3.8-flash-next-spark-idle/metrics-after.txt` |
| Cumulative acceptance since boot, at the start of the probe run (68.2%, 2.36 tokens per step, 32,643 draft steps), prefix-cache hit rate 96.8–97.5% | The counters of one `/metrics` snapshot (mixed traffic, mostly other users) | `uv run python tools/metrics-delta.py results/qwen3.8-flash-next-spark-probe/metrics-before.txt` (one file = since boot); live: `tools/spark_inspect.sh $SPARK_URL` |
| Prefix cache: 20k repeat 7.14 → 0.27 s; 6k repeat 2.39 → 0.25 s; shared system prompt 2.43 → 0.27 s; multi-turn 3.68 → 0.39 s | Same prompt twice, `max_tokens=1`, idle server | `uv run python tools/prefix_cache_scenarios.py --url ${SPARK_URL%/v1} --model qwen3.8-flash-next` → `results/qwen3.8-flash-next-spark-idle/prefix-cache-scenarios.log` |
| Server idle during the run (16 samples, max 1 running) | `vllm:num_requests_running` sampled every 15 s | `results/qwen3.8-flash-next-spark-idle/running-watch.txt` (written by `tools/spark-idle-run.sh`) |
| GSM8K 60/60, HumanEval 39/40 (1 truncated) | Same prompts and settings as the main study, `--limit 20` (idle) and `--limit 40` (probe) | `results/qwen3.8-flash-next-spark-idle/gsm8k.jsonl`, `results/qwen3.8-flash-next-spark-probe/{gsm8k,humaneval}.jsonl`; the probe's speed is discarded (its metrics delta shows 6.9M prompt tokens from another session) |
| Draft-vocabulary coverage 98.40% (all traces), 98.86% (Flash-Next output), per-task values, missed-token lists | Tokenize every saved reasoning trace + answer with the Flash-Next tokenizer, count ids outside `tools/draft_vocab_65536.npy` | `uv run --with tokenizers --with numpy python tools/draft-vocab-coverage.py --ids tools/draft_vocab_65536.npy --tokenizer <dir with the Flash-Next tokenizer.json>` → `tools/draft-vocab-coverage.out`; with `--runs qwen3.8-flash-next-spark-probe` → `results/qwen3.8-flash-next-spark-probe/draft-vocab-coverage.out` |
| Draft set = 64,902 ids below 65,536 plus 634 ids above; 6,666 distinct ids ever missed, top 40 = 27% of misses | `numpy.load` of the `.npy` and a count below/above 65,536; the missed-id lists in `tools/draft-vocab-coverage.out` | `uv run --with numpy python -c "import numpy as np; a=np.load('tools/draft_vocab_65536.npy'); print((a<65536).sum(), (a>=65536).sum())"` |
| Model identity, architecture, parameter counts, vLLM version and patches | `/v1/models` (`root` path with the snapshot), the Hugging Face config/card, and the image's boot log (version, patch count, `VLLM_MTP_DRAFT_VOCAB`) as reported by the image owner | `tools/spark_inspect.sh`; boot log on the Spark: `docker logs <container>` |
