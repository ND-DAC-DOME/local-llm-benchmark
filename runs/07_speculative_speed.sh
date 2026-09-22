#!/usr/bin/env bash
# Run 7: speed (decode + prefill) WITH speculative decoding for the configurations that ran without it.
# Speculative decoding is lossless, so only speed is re-measured; quality numbers come from the earlier runs.
#   a) Muse-Glimmer NVFP4 on vLLM + official DFlash drafter (meta-models/Muse-Glimmer-30B-assistant)
#      -> results/muse-glimmer-30b-nvfp4-a6000-dflash/
#   b) Muse-Glimmer Q4 GGUF on llama.cpp + unsloth's dflash-kquant.gguf drafter
#      -> results/muse-glimmer-30b-q4-dflash/
#   c) Qwen3.6-27B Q4 GGUF on llama.cpp with --spec-type draft-mtp: FAILS, the unsloth GGUF has no MTP layers (kept for the record)
#      -> results/qwen3.6-27b-q4-mtp/
# Needs both GPUs free (a on GPU 0, b and c on GPU 1, sequentially).
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; [ -f "$ROOT/config.env" ] || ROOT="$(dirname "$ROOT")"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a

HF=$HF_CACHE
BENCH_SPEED=".venv/bin/python bench.py --tasks speed --temperature 1.0 --top-p 0.95 --speed-concurrency 1"

# a) vLLM + DFlash
( export NAME=bench-vllm-muse-dflash GPU=${GPU_A:-0} PORT=8002
  MODEL=nvidia/Muse-Glimmer-30B-NVFP4 REVISION=47818374517751c48c55cde2621594926b1888b6 ALIAS=muse-glimmer-nvfp4 \
  REASONING_PARSER=muse_glimmer TOOL_PARSER=muse_glimmer KV_DTYPE=auto GPU_UTIL=0.90 MAX_LEN=32768 \
  EXTRA_ARGS='--trust-remote-code --mamba-cache-mode align --linear-backend marlin --compilation-config {"mode":0,"cudagraph_mode":"FULL"} --speculative-config {"method":"dflash","model":"meta-models/Muse-Glimmer-30B-assistant","revision":"e8192f3a8f617f74be2ce220360c89ef4789f39f","num_speculative_tokens":15}' \
  ./serve_vllm_nvfp4.sh && \
  $BENCH_SPEED --base-url http://localhost:$PORT/v1 --model muse-glimmer-nvfp4 --run-name muse-glimmer-30b-nvfp4-a6000-dflash --system "Reasoning strength: high"
  docker rm -f "$NAME" ) > results/run_07a.log 2>&1 &

# b) + c) llama.cpp
( source <(sed -n '/^IMAGE=/,/^LIMIT=/p' run_all.sh); export GPU=${GPU_B:-1}; PORT=8007
  D=$HF/hub/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/faa5b025c584459c13febfa5c59883516710ae39/dflash-kquant.gguf
  serve_spec() {  # serve_spec <name> <hf-repo:quant> <alias> <top_k> <extra llama-server args...>
    local name=$1 repo=$2 alias=$3 topk=$4; shift 4
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" --memory 8g --memory-swap 8g --gpus "device=$GPU" -p 127.0.0.1:$PORT:8080 \
      -v "$HF":/root/.cache/huggingface "$IMAGE" -hf "$repo" --no-mmproj -c 49152 --parallel 3 -ngl 99 --jinja \
      --top-k "$topk" --cache-ram 0 --alias "$alias" --host 0.0.0.0 --port 8080 "$@" >/dev/null
    for _ in $(seq 1 90); do
      [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && return 0
      [ "$(docker inspect "$name" --format '{{.State.Status}}')" = running ] || break; sleep 10
    done; echo "$name failed to start"; docker logs --tail 20 "$name"; return 1
  }
  serve_spec bench-llm-spec unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL muse-glimmer-30b-q4 64 \
      --spec-type draft-dflash --spec-draft-model "${D/$HF//root/.cache/huggingface}" && \
    $BENCH_SPEED --base-url http://localhost:$PORT/v1 --model muse-glimmer-30b-q4 --run-name muse-glimmer-30b-q4-dflash --system "Reasoning strength: high"
  docker rm -f bench-llm-spec
  serve_spec bench-llm-spec unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL qwen3.6-27b-q4 20 --spec-type draft-mtp && \
    $BENCH_SPEED --base-url http://localhost:$PORT/v1 --model qwen3.6-27b-q4 --run-name qwen3.6-27b-q4-mtp
  docker rm -f bench-llm-spec ) > results/run_07bc.log 2>&1 &
wait
echo ALL DONE
