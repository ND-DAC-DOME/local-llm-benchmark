#!/usr/bin/env bash
# Run 3: Qwen3.8-27B NVFP4 on vLLM, A6000, WITH MTP speculative decoding = the Spark recipe as closely as
# the A6000 allows (16k prefill batches, 131k context instead of 262k, Marlin FP4 emulation, patched lm_head).
# Needs a free 48 GB GPU. Results: results/qwen3.8-27b-nvfp4-a6000-mtp/ (speed) and
# results/qwen3.8-27b-nvfp4-a6000-mtp-32k/ (MMLU-Pro with a 32,768-token completion budget).
# The MTP run also repeats all quality tasks at the 12,288 budget, so the with/without-MTP versions are
# complete and comparable line by line (speculative decoding is lossless; scores should agree within noise).
cd "$(dirname "$0")/.."
export NAME=${NAME:-bench-vllm-qwen} GPU=${GPU:-1} PORT=${PORT:-8003}
GPU_UTIL=0.90 BATCHED_TOKENS=16384 MAX_LEN=131072 \
  EXTRA_ARGS='--speculative-config {"method":"mtp","num_speculative_tokens":3}' ./serve_vllm_nvfp4.sh
.venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model qwen3.8-27b \
  --run-name qwen3.8-27b-nvfp4-a6000-mtp --tasks speed --temperature 1.0 --top-p 0.95 --speed-concurrency 1
.venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model qwen3.8-27b \
  --run-name qwen3.8-27b-nvfp4-a6000-mtp-32k --tasks gsm8k,mmlu_pro,humaneval,speed --limit 200 --temperature 1.0 --top-p 0.95 \
  --max-tokens 32768 --concurrency 3
.venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model qwen3.8-27b \
  --run-name qwen3.8-27b-nvfp4-a6000-mtp --tasks gsm8k,mmlu_pro,humaneval --limit 200 --temperature 1.0 --top-p 0.95 \
  --max-tokens 12288 --concurrency 3
docker rm -f "$NAME"
