#!/usr/bin/env bash
# Run 5: Qwen3.8-27B NVFP4 as served on the DGX Spark (vLLM, docs/spark-qwen3.8-27b-recipe.yaml), measured
# over the network with the same script. Nothing is installed on the Spark. Speed numbers depend on the
# Spark being otherwise idle. Results: results/qwen3.8-27b-nvfp4-spark/
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; [ -f "$ROOT/config.env" ] || ROOT="$(dirname "$ROOT")"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a

SPARK=${SPARK:-$SPARK_URL}
[ -n "$SPARK" ] || { echo "set SPARK_URL in config.local.env (e.g. http://<spark-host>:18300/v1)"; exit 1; }
.venv/bin/python bench.py --base-url "$SPARK" --model qwen3.8-27b --run-name qwen3.8-27b-nvfp4-spark \
  --limit 200 --temperature 1.0 --top-p 0.95 --max-tokens 12288 --concurrency 3 --speed-concurrency 1
