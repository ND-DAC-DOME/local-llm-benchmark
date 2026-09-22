#!/usr/bin/env bash
# Run 6: Qwen3.6-27B UD-Q4_K_XL on llama.cpp, all tasks with a 32,768-token completion budget
# (pairs with run 3's 32k MMLU-Pro for Qwen3.8). Context per slot raised to 40k so prompt + 32k fits.
# Results: results/qwen3.6-27b-q4-32k/
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; [ -f "$ROOT/config.env" ] || ROOT="$(dirname "$ROOT")"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a

source <(sed -n '/^IMAGE=/,/^LIMIT=/p' run_all.sh)
NAME=bench-llm-32k; PORT=8004
source <(sed -n '/^serve() {/,/^}/p' run_all.sh | sed 's/-c 49152 --parallel 3/-c 122880 --parallel 3/')
serve unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL qwen3.6-27b-q4 20
.venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model qwen3.6-27b-q4 --run-name qwen3.6-27b-q4-32k \
  --tasks gsm8k,mmlu_pro,humaneval,speed --limit 200 --temperature 1.0 --top-p 0.95 --max-tokens 32768 --concurrency 3 --speed-concurrency 1
docker rm -f "$NAME"
