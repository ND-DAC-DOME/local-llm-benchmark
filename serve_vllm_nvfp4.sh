#!/usr/bin/env bash
# Serve the exact model the Spark runs (unsloth/Qwen3.8-27B-NVFP4, commit f0b7c9e) with vLLM on the A6000.
# Adapted from docs/spark-qwen3.8-27b-recipe.yaml. Differences from the Spark recipe, all forced by hardware:
#   - image $VLLM_IMAGE from config.env (vLLM 0.29.0, x86; the Spark image is ARM64)          - no --load-format instanttensor
#   - NVFP4 runs through vLLM's Marlin weight-only FP4 emulation (A6000 = Ampere, no native FP4)
#   - patches/humming_utils.py: 3-line fix for a vLLM 0.29.0 crash on the FP8 lm_head in this path
#   - no MTP speculative decoding, no vision encoder, max-model-len 32k, max-num-batched-tokens 4096
#     (16384 OOMs on a 12k-token prefill next to the other server on GPU 0): all VRAM-driven
#   - --memory 16g cap so a leak cannot take the host down
# Muse-Glimmer NVFP4 (nvidia card recipe): MODEL=nvidia/Muse-Glimmer-30B-NVFP4 REVISION=4781837... ALIAS=muse-glimmer-nvfp4 \
#   REASONING_PARSER=muse_glimmer TOOL_PARSER=muse_glimmer KV_DTYPE=auto EXTRA_ARGS='--trust-remote-code --mamba-cache-mode align'
set -euo pipefail
cd "$(dirname "$0")"
NAME=${NAME:-bench-vllm}; GPU=${GPU:-0}; PORT=${PORT:-8002}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --memory 16g --memory-swap 16g --shm-size 4g --gpus "device=$GPU" \
  -p 127.0.0.1:$PORT:8000 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 ${EXTRA_ENV:-} \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$PWD/patches/humming_utils.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/humming_utils.py:ro \
  "$VLLM_IMAGE" "${MODEL:-unsloth/Qwen3.8-27B-NVFP4}" --revision "${REVISION:-f0b7c9e722f5565102fff8481c99e4d86ae099c7}" \
  --served-model-name "${ALIAS:-qwen3.8-27b}" --gpu-memory-utilization "${GPU_UTIL:-0.59}" --max-model-len "${MAX_LEN:-32768}" \
  --max-num-seqs 3 --max-num-batched-tokens "${BATCHED_TOKENS:-4096}" --kv-cache-dtype "${KV_DTYPE:-fp8}" --attention-backend flashinfer \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-prefix-caching ${EXTRA_ARGS:-} >/dev/null
for _ in $(seq 1 90); do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:$PORT/v1/models)" = 200 ] && { echo "vLLM up on :$PORT"; exit 0; }
  [ "$(docker inspect "$NAME" --format '{{.State.Status}}')" = running ] || break
  sleep 10
done
echo "vLLM failed to start:"; docker logs --tail 30 "$NAME"; exit 1
