#!/usr/bin/env bash
# Run 1: Muse-Glimmer-30B vs Qwen3.6-27B, both unsloth UD-Q4_K_XL GGUF on llama.cpp, GPU 0.
# Full pipeline (serve one model at a time, run all tasks) is ../run_all.sh. Speed-only re-measure: ../speed_only.sh
cd "$(dirname "$0")/.." && exec ./run_all.sh
