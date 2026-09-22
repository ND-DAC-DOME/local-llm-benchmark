#!/usr/bin/env bash
# Run 4: Muse-Glimmer-30B NVFP4 (nvidia/Muse-Glimmer-30B-NVFP4, ModelOpt AutoQuant: mixed NVFP4/FP8/BF16)
# on vLLM, A6000, following the NVIDIA model card's vLLM recipe. Muse has no MTP head; its speculative
# decoder (DFlash) is not part of the NVIDIA vLLM recipe, so none is used.
# A6000 workarounds: --linear-backend marlin (Ampere cannot run the FP8 W8A8 layers; Marlin does them
# weight-only), torch.compile disabled (Inductor crashes on this model; CUDA graphs kept).
# Results: results/muse-glimmer-30b-nvfp4-a6000/
cd "$(dirname "$0")/.."
export NAME=${NAME:-bench-vllm} GPU=${GPU:-0} PORT=${PORT:-8002}
MODEL=nvidia/Muse-Glimmer-30B-NVFP4 REVISION=47818374517751c48c55cde2621594926b1888b6 ALIAS=muse-glimmer-nvfp4 \
  REASONING_PARSER=muse_glimmer TOOL_PARSER=muse_glimmer KV_DTYPE=auto GPU_UTIL=${GPU_UTIL:-0.59} \
  EXTRA_ARGS='--trust-remote-code --mamba-cache-mode align --linear-backend marlin --compilation-config {"mode":0,"cudagraph_mode":"FULL"}' \
  ./serve_vllm_nvfp4.sh
.venv/bin/python bench.py --base-url http://localhost:$PORT/v1 --model muse-glimmer-nvfp4 \
  --run-name muse-glimmer-30b-nvfp4-a6000 --limit 200 --system "Reasoning strength: high" \
  --temperature 1.0 --top-p 0.95 --max-tokens 12288 --concurrency 3 --speed-concurrency 1
docker rm -f "$NAME"
