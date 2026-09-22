#!/usr/bin/env bash
# Re-measure decode + prefill speed for an already-benchmarked llama.cpp run (same serve flags as run_all.sh).
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; [ -f "$ROOT/config.env" ] || ROOT="$(dirname "$ROOT")"
set -a; . "$ROOT/config.env"; [ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"; set +a

source <(sed -n '/^IMAGE=/,/^LIMIT=/p' run_all.sh)
source <(sed -n '/^serve() {/,/^}/p' run_all.sh)
serve "$1" "$2" "$3"
.venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model "$2" --run-name "$2" --tasks speed \
  --system "${4:-}" --temperature 1.0 --top-p 0.95 --speed-concurrency 1
docker rm -f "$NAME" >/dev/null
