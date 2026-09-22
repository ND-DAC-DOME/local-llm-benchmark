## Benchmarks used in our local comparison

| Benchmark | What it measures | Dataset (the one we run) | Paper | Code |
|---|---|---|---|---|
| GSM8K | Grade-school math word problems (1,319 test items) | https://huggingface.co/datasets/openai/gsm8k | https://arxiv.org/abs/2110.14168 | https://github.com/openai/grade-school-math |
| MMLU-Pro | 10-option multiple choice across 14 subjects (~12k items) | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | https://arxiv.org/abs/2406.01574 | https://github.com/TIGER-AI-Lab/MMLU-Pro |
| HumanEval | 164 Python functions, graded by running unit tests | https://huggingface.co/datasets/openai/openai_humaneval | https://arxiv.org/abs/2107.03374 | https://github.com/openai/human-eval |
| GPQA Diamond | 198 graduate-level science questions, 4 options | https://huggingface.co/datasets/hendrydong/gpqa_diamond_mc (ungated mirror) | https://arxiv.org/abs/2311.12022 | https://github.com/idavidrein/gpqa |
| LiveCodeBench v6 | 175 competitive-programming problems (contests Jan–Apr 2025), graded by running all tests | https://huggingface.co/datasets/livecodebench/code_generation_lite (`test6.jsonl`) | https://arxiv.org/abs/2403.07974 | https://github.com/LiveCodeBench/LiveCodeBench |

**Papers**

- GSM8K: "Training Verifiers to Solve Math Word Problems" (OpenAI, 2021).
- MMLU-Pro: "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark" (TIGER-Lab, 2024).
- HumanEval: "Evaluating Large Language Models Trained on Code" (OpenAI, 2021). This is the Codex paper; HumanEval was introduced in it.
- GPQA: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" (2023).
- LiveCodeBench: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code" (2024).

**How we run them**

- GSM8K and MMLU-Pro: a fixed random sample of 200 items from the test split (seed 1234), the same items for every model. HumanEval: all 164 problems. GPQA Diamond: all 198. LiveCodeBench v6: all 175.
- Zero-shot, one chat turn per item. The model is asked to end with `Final answer: <number>` (GSM8K), `Answer: <letter>` (MMLU-Pro, GPQA), or a single Python code block (HumanEval, LiveCodeBench).
- GSM8K, MMLU-Pro and GPQA are graded by exact match against the gold answer. HumanEval and LiveCodeBench are graded by executing the problem's tests against the model's code; a problem counts only if every test passes. LiveCodeBench is scored twice: with the standard library only (strict) and with numpy/scipy/numba available as on the real AtCoder judge (judge env, the headline).
- A response that hits the token limit before giving an answer counts as wrong and is reported as "not finished". Two budgets were used, 12,288 and 32,768 tokens, always the same for every model in a comparison; the budget turned out to decide the ranking, so both are reported.
- Same sampling for every model: temperature 1.0, top_p 0.95 (the recommendation in both vendors' model cards); top_k is each vendor's own recommendation.
- Weights are 4-bit: unsloth `UD-Q4_K_XL` GGUF on llama.cpp for Muse and Qwen3.6; unsloth NVFP4 on vLLM for Qwen3.8 (the checkpoint the DGX Spark serves); NVIDIA's NVFP4 Muse for the quantization check. See `MODELS.md`.
- Every prompt, reasoning trace, response and verdict is saved to `results/<run>/<task>.jsonl` so scores can be audited.

**Caveats**

- Our prompts and answer parsers are our own, so these scores are not directly comparable to public leaderboards (which use different prompt formats, few-shot examples and larger budgets). They are meant for comparing models against each other under identical conditions.
- With ~200 items the 95% margin of error is roughly ±3 to ±6 percentage points, depending on the score. Measured directly: the same model run three times spread by ±1.5 points. Differences inside that margin should be read as a tie.
- GSM8K and HumanEval are close to the ceiling for models of this class (96-99%), so they work as sanity checks. MMLU-Pro, GPQA Diamond and LiveCodeBench have room to separate models; AIME was considered and dropped (vendors report 94-95%).
- None of these is an agentic software-engineering test. For SWE-bench numbers published by the vendors, see `swe-bench-cards.md`.

Links checked on 2026-09-17 (first three) and 2026-09-22 (GPQA, LiveCodeBench): all resolve, and the arXiv titles match.
