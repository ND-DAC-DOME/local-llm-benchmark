`humming_utils.py` is vLLM 0.29.0's `vllm/model_executor/layers/quantization/utils/humming_utils.py` (Apache-2.0,
© vLLM contributors) with the small fix shown in `humming_utils.patch`: `ParallelLMHead` has no
`output_partition_sizes` / `has_bias`, which crashes weight post-processing for checkpoints with a quantized
`lm_head` (e.g. `unsloth/Qwen3.8-27B-NVFP4`) on GPUs that take the Marlin path (pre-Blackwell).
`serve_vllm_nvfp4.sh` bind-mounts the patched file over the one in the image. It only applies to the pinned image.
