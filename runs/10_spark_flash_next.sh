#!/usr/bin/env bash
# Run 10: Qwen3.8-Flash-Next (nvidia/Qwen3.8-Flash-Next-NVFP4, 125B total / 6B active MoE) as served on the
# DGX Spark by the qwen38-flash-next-spark image (vLLM 0.31 + Spark patches, MTP with 2 draft tokens and a
# reduced 65k draft vocabulary), model id qwen3.8-flash-next, measured over the network. Nothing is installed
# on the Spark. This is a different model from run 5's Qwen3.8-27B, not the same model on a new server.
#
# The idle run is what results.md reports: tools/spark-idle-run.sh refuses to start unless the server is idle,
# samples its request count every 15 s (results/<run>/running-watch.txt) and snapshots /metrics before and after,
# so the MTP acceptance and prefix-cache figures are this run's own.
#   -> results/qwen3.8-flash-next-spark-idle/   (speed + prefill, GSM8K 20, prefix-cache scenarios)
#
# The probe run ran first with the same settings on more items, but another session used the server during it
# (its metrics delta shows 6.9M prompt tokens for 80 items); keep its quality scores and traces, not its speed:
#   -> results/qwen3.8-flash-next-spark-probe/  (speed [discard], GSM8K 40, HumanEval 40)
#   tools/spark-idle-run.sh qwen3.8-flash-next-spark-probe --tasks speed,gsm8k,humaneval --limit 40 \
#       --temperature 1.0 --top-p 0.95 --max-tokens 12288 --concurrency 1 --speed-concurrency 1
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. ./env.sh
[ -n "${SPARK_URL:-}" ] || { echo "set SPARK_URL (and SPARK_API_KEY, SPARK_CA_BUNDLE) in .env"; exit 1; }

RUN=${RUN:-qwen3.8-flash-next-spark-idle}
tools/spark-idle-run.sh "$RUN" --tasks speed,gsm8k --limit 20 \
  --temperature 1.0 --top-p 0.95 --max-tokens 12288 --concurrency 1 --speed-concurrency 1

# Prefix-cache behaviour on the same server (five usage patterns; needs the server idle as well).
uv run python tools/prefix_cache_scenarios.py --url "${SPARK_URL%/v1}" --model qwen3.8-flash-next \
  | tee "results/$RUN/prefix-cache-scenarios.log"

# Share of the generated tokens that fall inside the drafter's reduced vocabulary (tokenizer kept outside the
# repo; get it from the Flash-Next checkpoint's tokenizer.json). Writes the per-run report next to the results.
TOK=${FLASH_NEXT_TOKENIZER:-$HOME/.cache/flash-next-tokenizer}
if [ -f "$TOK/tokenizer.json" ]; then
  uv run --with tokenizers --with numpy python tools/draft-vocab-coverage.py \
    --ids tools/draft_vocab_65536.npy --tokenizer "$TOK" --runs "$RUN" | tee "results/$RUN/draft-vocab-coverage.out"
else
  echo "tokenizer not found at $TOK; skipping draft-vocabulary coverage"
fi
