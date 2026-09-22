## SWE-bench: Muse-Glimmer-30B vs Qwen3.6-27B (vendor-published numbers)

| Model / Benchmark | Meta's card | Qwen's cards (3.6 / 3.8) |
|---|---|---|
| Muse-Glimmer-30B, SWE-Bench Verified | 76.0 | not reported |
| Muse-Glimmer-30B, SWE-Bench Pro | 51.2 | not reported |
| Qwen3.6-27B, SWE-Bench Verified | 77.2 | 77.2 |
| Qwen3.6-27B, SWE-Bench Pro | 50.2 | 53.5 |
| Gemma4-31B, SWE-Bench Verified | 66.6 | 52.0 |
| Gemma4-31B, SWE-Bench Pro | 36.9 | 35.7 |
| Qwen3.8-27B, SWE-Bench Pro | not reported | 61.7 (its own card; same card lists Muse-Glimmer at 51.2 and Qwen3.6 at 53.5) |

**Sources**

- Muse-Glimmer-30B model card (Meta): https://huggingface.co/meta-models/Muse-Glimmer-30B
  - Raw text (table at lines 90–102, "Agentic Coding" rows): https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/README.md
- Qwen3.6-27B model card (Qwen): https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3.8-27B model card (Qwen): https://huggingface.co/Qwen/Qwen3.8-27B (SWE-Bench Pro only; no Verified row)
  - Raw text ("Benchmark Results → Language → Coding Agent"; methodology notes near line 270): https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/README.md

**Notes**

- All numbers are self-reported by the vendors; none were reproduced by us.
- Reasoning modes as labelled in Meta's card: Muse-Glimmer "High Reasoning", Qwen3.6-27B and Gemma4-31B "Thinking Mode". Qwen's card does not label the mode in its table header.
- Neither card states the precision used for these tables. They are presumably full-precision (BF16) results: Meta's card treats "Full Precision" as the baseline and separately reports the loss of its own 4-bit quants (0.2% for K-Quant-Dynamic, 1.0% for K-Quant-17GB, averaged over 15 benchmarks). Qwen's card does not mention quantization at all.
- These are NOT numbers for quantized models. Our own local tests run 4-bit weights (unsloth `UD-Q4_K_XL` GGUF for Muse and Qwen3.6, unsloth NVFP4 for Qwen3.8, NVIDIA NVFP4 for the Muse quantization check), none of which are Meta's K-Quants, so Meta's degradation figures do not directly apply to them either. Our own check found no quantization effect for Muse (Q4 GGUF and NVFP4 score the same).
- Each vendor uses its own agent scaffold. Qwen states it uses an internal scaffold (bash + file-edit tools), temp 1.0, top_p 0.95, 200K context, and that it corrected some problematic tasks in the public SWE-bench Pro set. The numbers from the two cards are therefore not directly comparable.
- The cards agree on Qwen's SWE-Bench Verified (77.2) and disagree on Qwen's SWE-Bench Pro (50.2 vs 53.5) and on Gemma4-31B's Verified (66.6 vs 52.0).
- The Muse-Glimmer number exists only as measured by Meta itself.
- Values extracted from the raw README text of the Meta and Qwen3.6 cards on 2026-09-17 and of the Qwen3.8 card on 2026-09-22.
