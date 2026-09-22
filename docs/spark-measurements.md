# How every Spark number in `results.md` was obtained

Nothing was installed or changed on the Spark. All measurements are network requests to its vLLM server
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
