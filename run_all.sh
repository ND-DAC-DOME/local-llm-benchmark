#!/usr/bin/env bash
# Muse-Glimmer-30B vs Qwen3.6-27B, same quant (UD-Q4_K_XL), same prompts, same sampling
# (temp 1.0 / top_p 0.95 are both vendors' recommendation; top_k is each vendor's own).
# Models are served one at a time on GPU 0 behind a RAM-capped container, port 8001.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; [ -f "$ROOT/config.env" ] || ROOT="$(dirname "$ROOT")"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a


IMAGE=$LLAMACPP_IMAGE
CACHE=$HF_CACHE
NAME=bench-llm
PORT=8001
LIMIT=${LIMIT:-200}

serve() {  # serve <hf-repo:quant> <alias> <top_k>
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --memory 8g --memory-swap 8g --gpus "device=${GPU:-0}" \
    -p 127.0.0.1:$PORT:8080 -v "$CACHE":/root/.cache/huggingface "$IMAGE" \
    -hf "$1" --no-mmproj -c 49152 --parallel 3 -ngl 99 --jinja --top-k "$3" --cache-ram 0 \
    --alias "$2" --host 0.0.0.0 --port 8080 >/dev/null
  for _ in $(seq 1 180); do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:$PORT/health)" = 200 ] && return 0
    [ "$(docker inspect "$NAME" --format '{{.State.Status}}')" = running ] || break
    sleep 10
  done
  echo "server for $2 failed to start:"; docker logs --tail 20 "$NAME"; return 1
}

bench() {  # bench <alias> <system-prompt>
  .venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model "$1" --run-name "$1" \
    --limit "$LIMIT" --system "$2" --temperature 1.0 --top-p 0.95 --max-tokens 12288 \
    --concurrency 3 --speed-concurrency 1
}

serve unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL muse-glimmer-30b-q4 64
bench muse-glimmer-30b-q4 "Reasoning strength: high"

serve unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL qwen3.6-27b-q4 20
bench qwen3.6-27b-q4 ""   # thinking mode is Qwen3.6's default

docker rm -f "$NAME" >/dev/null
echo ALL DONE
