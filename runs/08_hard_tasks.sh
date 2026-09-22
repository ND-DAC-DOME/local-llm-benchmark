#!/usr/bin/env bash
# Run 8: harder tasks that still separate these models: GPQA Diamond (198 items) and LiveCodeBench v6
# (175 problems, 2025-01..04), 32,768-token budget for every model, each model in its fastest lossless
# configuration (speculative decoding does not change answers):
#   GPU 0: Qwen3.8-27B NVFP4 + MTP on vLLM          -> results/qwen3.8-27b-nvfp4-a6000-mtp-32k/
#   GPU 1: Muse-Glimmer Q4 GGUF + DFlash, llama.cpp -> results/muse-glimmer-30b-q4-dflash-32k/
# Qwen3.6 was left out of this run by decision (it is not the model served on the Spark, and without MTP in its
# GGUF it would take the longest).
# AIME was considered and dropped: vendors report 94-95% for these models (saturated).
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; [ -f "$ROOT/config.env" ] || ROOT="$(dirname "$ROOT")"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a

HF=$HF_CACHE
TASKS=${TASKS:-gpqa_diamond,livecodebench}
SPEED=".venv/bin/python bench.py --tasks speed --temperature 1.0 --top-p 0.95 --speed-concurrency 1"
BENCH=".venv/bin/python bench.py --tasks $TASKS --limit 0 --temperature 1.0 --top-p 0.95 --max-tokens 32768"

( export NAME=bench-vllm-qwen GPU=${GPU_A:-0} PORT=8003
  GPU_UTIL=0.90 BATCHED_TOKENS=16384 MAX_LEN=131072 \
    EXTRA_ARGS='--speculative-config {"method":"mtp","num_speculative_tokens":3}' ./serve_vllm_nvfp4.sh && \
  $BENCH --base-url http://localhost:$PORT/v1 --model qwen3.8-27b --run-name qwen3.8-27b-nvfp4-a6000-mtp-32k --concurrency 3
  $SPEED --base-url http://localhost:$PORT/v1 --model qwen3.8-27b --run-name qwen3.8-27b-nvfp4-a6000-mtp-32k   # same server, GPU idle
  docker rm -f "$NAME" ) > results/run_08_qwen3.8.log 2>&1 &

( source <(sed -n '/^IMAGE=/,/^LIMIT=/p' run_all.sh); GPU=${GPU_B:-1}; PORT=8007; NAME=bench-llm-hard
  serve_gguf() {  # serve_gguf <hf-repo:quant> <alias> <top_k> <ctx> <slots> <extra args...>
    local repo=$1 alias=$2 topk=$3 ctx=$4 slots=$5; shift 5
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --memory 8g --memory-swap 8g --gpus "device=$GPU" -p 127.0.0.1:$PORT:8080 \
      -v "$HF":/root/.cache/huggingface "$IMAGE" -hf "$repo" --no-mmproj -c "$ctx" --parallel "$slots" -ngl 99 --jinja \
      --top-k "$topk" --cache-ram 0 --alias "$alias" --host 0.0.0.0 --port 8080 "$@" >/dev/null
    for _ in $(seq 1 90); do
      [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && return 0
      [ "$(docker inspect "$NAME" --format '{{.State.Status}}')" = running ] || break; sleep 10
    done; echo "$alias failed to start"; docker logs --tail 20 "$NAME"; return 1
  }
  D=/root/.cache/huggingface/hub/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/faa5b025c584459c13febfa5c59883516710ae39/dflash-kquant.gguf
  # Muse is a dense transformer (large KV cache): 2 slots x 36k context is what fits in 48 GB next to the drafter.
  serve_gguf unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL muse-glimmer-30b-q4 64 73728 2 --spec-type draft-dflash --spec-draft-model "$D" && \
    $BENCH --base-url http://localhost:$PORT/v1 --model muse-glimmer-30b-q4 --run-name muse-glimmer-30b-q4-dflash-32k \
      --system "Reasoning strength: high" --concurrency 2
  $SPEED --base-url http://localhost:$PORT/v1 --model muse-glimmer-30b-q4 --run-name muse-glimmer-30b-q4-dflash-32k --system "Reasoning strength: high"
  docker rm -f "$NAME" ) > results/run_08_gguf.log 2>&1 &
wait
echo ALL DONE
