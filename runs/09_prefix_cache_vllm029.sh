#!/usr/bin/env bash
# Run 9: does vLLM's prefix cache work for Qwen3.8-27B NVFP4 (a hybrid GDN + attention model) with MTP on?
# On the Spark (vLLM 0.26.1 + MTP) it has no effect: repeating a 6k..129k-token prompt is never faster (tools/).
# This repeats the check on vLLM 0.29.0 (A6000), same recipe, two conditions: MTP on, then MTP off.
# If caching works here, the Spark's fix is a vLLM upgrade; if not, it is a vLLM limitation for this model.
# Cost: ~20 min on one GPU; only max_tokens=1 requests. Output: results/run_09_prefix_cache.log
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./config.env; [ -f ./config.local.env ] && . ./config.local.env; set +a
export NAME=bench-vllm-cache GPU=${GPU:-0} PORT=${PORT:-8003}
for cond in "mtp-on:--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3}" "mtp-off:"; do
  label=${cond%%:*}; extra=${cond#*:}
  echo "=== condition: $label"
  GPU_UTIL=0.90 BATCHED_TOKENS=16384 MAX_LEN=131072 EXTRA_ARGS="$extra" ./serve_vllm_nvfp4.sh || { echo "server failed"; continue; }
  docker logs "$NAME" 2>&1 | grep -iE "prefix.cach|retention|mamba cache|block size" | sed 's/.*\] //' | cut -c1-200
  .venv/bin/python tools/check_prefix_cache.py --url http://localhost:$PORT --model qwen3.8-27b --pairs 4
  .venv/bin/python tools/prefix_cache_scenarios.py --url http://localhost:$PORT --model qwen3.8-27b
  docker rm -f "$NAME" >/dev/null
done
echo "PREFIX CACHE CHECK DONE"
