#!/usr/bin/env bash
# Run 2: Qwen3.8-27B NVFP4 (the Spark's exact checkpoint) on vLLM, A6000, WITHOUT MTP speculative decoding.
# This was the configuration that fit next to another server on GPU 0 (28 GB free): no MTP, no vision
# encoder, 32k context, 4k prefill batches. Results: results/qwen3.8-27b-nvfp4-a6000/
cd "$(dirname "$0")/.."
GPU=${GPU:-0} PORT=${PORT:-8002} GPU_UTIL=${GPU_UTIL:-0.59} BATCHED_TOKENS=4096 MAX_LEN=32768 ./serve_vllm_nvfp4.sh
.venv/bin/python bench.py --base-url http://localhost:${PORT:-8002}/v1 --model qwen3.8-27b \
  --run-name qwen3.8-27b-nvfp4-a6000 --limit 200 --temperature 1.0 --top-p 0.95 --max-tokens 12288 \
  --concurrency 3 --speed-concurrency 1
docker rm -f "${NAME:-bench-vllm}"
