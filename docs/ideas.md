# Parked experiments

## Union-of-truncated re-run at a large budget (not run)

Build one item set = every GPQA Diamond / LiveCodeBench item that either model truncated at 32k (97 items: 31 + 66;
every item Muse truncated, Qwen3.8 truncated too), and have both models answer it at a large budget (64k or 128k).
Answers "can Qwen3.8 solve these when not cut off, and at what token cost". Estimated 9 h (64k, LCB only) to
28 h (128k, both tasks) on the A6000s, dominated by Qwen3.8. Parked until the team says what answer latency their
use tolerates, since that fixes the budget that matters; 128k answers take 27 min on the A6000 and >1 h on the Spark.
`bench.py --retry-truncated-from <run>` exists for the per-model variant.
