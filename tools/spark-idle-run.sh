#!/usr/bin/env bash
# One clean benchmark run against the Spark, with proof that nobody else was using it.
#
#   tools/spark-idle-run.sh <run-name> [bench.py args...]
#
# Snapshots /metrics before and after, samples vllm:num_requests_running every 15 s during the run
# (results/<run>/running-watch.txt), then prints the run's own acceptance and prefix-cache numbers
# (tools/metrics-delta.py) and the speed line. Refuses to start unless the server is idle.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. ./env.sh
RUN="${1:?run name}"; shift
OUT="results/$RUN"; mkdir -p "$OUT"
M="${SPARK_URL%/v1}/metrics"
AUTH=(-H "Authorization: Bearer ${SPARK_API_KEY:-}")

metrics() { curl -s -m 10 "${AUTH[@]}" "$M"; }
snap() { metrics | grep -E '^vllm:(spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|spec_decode_num_accepted_tokens_per_pos_total|prefix_cache_(hits|queries)_total|generation_tokens_total|prompt_tokens_total)'; }
running() { metrics | grep -E '^vllm:num_requests_running\{' | sed -E 's/.* //'; }

for _ in 1 2 3; do
  r="$(running)"
  [ "${r%.*}" = 0 ] || { echo "!! server busy: num_requests_running=$r"; exit 1; }
  sleep 3
done
echo ">> server idle, starting $RUN: $*"

snap > "$OUT/metrics-before.txt"
( while true; do echo "$(date +%H:%M:%S) $(running)"; sleep 15; done > "$OUT/running-watch.txt" ) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

.venv/bin/python bench.py --base-url "$SPARK_URL" --model qwen3.8-flash-next --run-name "$RUN" "$@" > "$OUT/bench.log" 2>&1 || echo "!! bench.py exited with $?"

kill $WATCH 2>/dev/null; trap - EXIT
snap > "$OUT/metrics-after.txt"

echo "=== concurrency watch"
awk '{ n++; if ($2 > max) max = $2; if ($2 > 1) busy++ } END { printf "samples %d, max running %s, samples above 1: %d\n", n, max, busy + 0 }' "$OUT/running-watch.txt"
echo "=== this run's share of the server"
.venv/bin/python tools/metrics-delta.py "$OUT/metrics-before.txt" "$OUT/metrics-after.txt"
echo "=== bench summary lines"
grep -E '^(speed|gsm8k|humaneval|mmlu_pro|gpqa_diamond|livecodebench) ' "$OUT/bench.log" || tail -5 "$OUT/bench.log"
